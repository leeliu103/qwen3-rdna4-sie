#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string prompt;
    std::string output;
    int n_predict = 16;
    int logits_top_k = 20;
};

struct TopLogit {
    llama_token token;
    float logit;
};

struct LogitsDump {
    llama_pos position;
    std::vector<TopLogit> topk_logits;
    llama_token greedy_next_token;
};

struct DecodeDump {
    int step;
    LogitsDump logits;
};

class LlamaBatch {
public:
    explicit LlamaBatch(int32_t capacity) : batch_(llama_batch_init(capacity, 0, 1)), capacity_(capacity) {
        if (capacity <= 0 ||
                batch_.token == nullptr ||
                batch_.pos == nullptr ||
                batch_.n_seq_id == nullptr ||
                batch_.seq_id == nullptr ||
                batch_.seq_id[0] == nullptr ||
                batch_.logits == nullptr) {
            llama_batch_free(batch_);
            throw std::runtime_error("failed to allocate llama batch");
        }
        batch_.n_tokens = 0;
    }

    ~LlamaBatch() {
        llama_batch_free(batch_);
    }

    LlamaBatch(const LlamaBatch &) = delete;
    LlamaBatch & operator=(const LlamaBatch &) = delete;

    llama_batch & get() {
        return batch_;
    }

    void clear() {
        batch_.n_tokens = 0;
    }

    void add(llama_token token, llama_pos pos, bool logits) {
        if (batch_.n_tokens >= capacity_) {
            throw std::runtime_error("llama batch capacity exceeded");
        }
        const int32_t i = batch_.n_tokens;
        batch_.token[i] = token;
        batch_.pos[i] = pos;
        batch_.n_seq_id[i] = 1;
        batch_.seq_id[i][0] = 0;
        batch_.logits[i] = logits ? 1 : 0;
        batch_.n_tokens++;
    }

private:
    llama_batch batch_;
    int32_t capacity_;
};

class LlamaBackend {
public:
    LlamaBackend() {
        llama_backend_init();
    }

    ~LlamaBackend() {
        llama_backend_free();
    }

    LlamaBackend(const LlamaBackend &) = delete;
    LlamaBackend & operator=(const LlamaBackend &) = delete;
};

class LlamaModel {
public:
    LlamaModel(const std::string & path, const llama_model_params & params)
        : model_(llama_model_load_from_file(path.c_str(), params)) {
        if (model_ == nullptr) {
            throw std::runtime_error("unable to load model: " + path);
        }
    }

    ~LlamaModel() {
        llama_model_free(model_);
    }

    LlamaModel(const LlamaModel &) = delete;
    LlamaModel & operator=(const LlamaModel &) = delete;

    llama_model * get() {
        return model_;
    }

private:
    llama_model * model_ = nullptr;
};

class LlamaContext {
public:
    LlamaContext(llama_model * model, const llama_context_params & params)
        : ctx_(llama_init_from_model(model, params)) {
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to create llama context");
        }
    }

    ~LlamaContext() {
        llama_free(ctx_);
    }

    LlamaContext(const LlamaContext &) = delete;
    LlamaContext & operator=(const LlamaContext &) = delete;

    llama_context * get() {
        return ctx_;
    }

private:
    llama_context * ctx_ = nullptr;
};

[[noreturn]] void fail_usage(const std::string & message) {
    throw std::runtime_error(message + "\nusage: llama-golden-dump --model PATH --prompt TEXT --output PATH [--n-predict N>0] [--logits-top-k N>0]");
}

int parse_nonnegative_int(const char * text, const std::string & name) {
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 || value > std::numeric_limits<int>::max()) {
        fail_usage(name + " must be a nonnegative integer");
    }
    return static_cast<int>(value);
}

int parse_positive_int(const char * text, const std::string & name) {
    const int value = parse_nonnegative_int(text, name);
    if (value <= 0) {
        fail_usage(name + " must be greater than 0");
    }
    return value;
}

Args parse_args(int argc, char ** argv) {
    Args args;
    bool have_model = false;
    bool have_prompt = false;
    bool have_output = false;
    bool have_n_predict = false;
    bool have_logits_top_k = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string & name) -> const char * {
            if (i + 1 >= argc) {
                fail_usage(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--model") {
            if (have_model) {
                fail_usage("--model was provided more than once");
            }
            args.model = require_value(arg);
            have_model = true;
        } else if (arg == "--prompt") {
            if (have_prompt) {
                fail_usage("--prompt was provided more than once");
            }
            args.prompt = require_value(arg);
            have_prompt = true;
        } else if (arg == "--output") {
            if (have_output) {
                fail_usage("--output was provided more than once");
            }
            args.output = require_value(arg);
            have_output = true;
        } else if (arg == "--n-predict") {
            if (have_n_predict) {
                fail_usage("--n-predict was provided more than once");
            }
            args.n_predict = parse_positive_int(require_value(arg), arg);
            have_n_predict = true;
        } else if (arg == "--logits-top-k") {
            if (have_logits_top_k) {
                fail_usage("--logits-top-k was provided more than once");
            }
            args.logits_top_k = parse_positive_int(require_value(arg), arg);
            have_logits_top_k = true;
        } else {
            fail_usage("unknown argument: " + arg);
        }
    }

    if (!have_model) {
        fail_usage("--model is required");
    }
    if (!have_prompt) {
        fail_usage("--prompt is required");
    }
    if (!have_output) {
        fail_usage("--output is required");
    }

    return args;
}

std::string json_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size() + 2);

    for (const unsigned char c : text) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                static const char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[(c >> 4) & 0x0f];
                escaped += hex[c & 0x0f];
            } else {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }

    return escaped;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int32_t needed = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, false);
    if (needed == INT32_MIN) {
        throw std::runtime_error("prompt tokenization overflowed");
    }

    const int32_t n_tokens = needed < 0 ? -needed : needed;
    if (n_tokens == 0) {
        throw std::runtime_error("prompt tokenization returned zero tokens");
    }

    std::vector<llama_token> tokens(n_tokens);
    const int32_t actual = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()), tokens.data(), n_tokens, true, false);
    if (actual < 0) {
        throw std::runtime_error("failed to tokenize prompt");
    }
    if (actual == 0) {
        throw std::runtime_error("prompt tokenization returned zero tokens");
    }
    tokens.resize(actual);
    return tokens;
}

std::vector<TopLogit> collect_topk_logits(const float * logits, int32_t n_vocab, int top_k) {
    if (logits == nullptr) {
        throw std::runtime_error("logits are unavailable");
    }
    if (n_vocab <= 0) {
        throw std::runtime_error("vocabulary size is invalid");
    }

    std::vector<TopLogit> all;
    all.reserve(static_cast<size_t>(n_vocab));
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (!std::isfinite(logits[token])) {
            throw std::runtime_error("logit for token " + std::to_string(token) + " is not finite");
        }
        all.push_back({ token, logits[token] });
    }

    const size_t want = std::min(static_cast<size_t>(top_k), all.size());
    auto better = [](const TopLogit & a, const TopLogit & b) {
        if (a.logit == b.logit) {
            return a.token < b.token;
        }
        return a.logit > b.logit;
    };

    std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(want), all.end(), better);
    all.resize(want);
    return all;
}

llama_token greedy_token_from_topk(const std::vector<TopLogit> & topk) {
    if (topk.empty()) {
        throw std::runtime_error("top-k logits are empty");
    }
    return topk.front().token;
}

void decode_batch(llama_context * ctx, llama_batch & batch) {
    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }
}

void write_topk(std::ostream & out, const std::vector<TopLogit> & topk, const std::string & indent) {
    out << "[\n";
    for (size_t i = 0; i < topk.size(); ++i) {
        out << indent << "{\"token\": " << topk[i].token << ", \"logit\": " << topk[i].logit << "}";
        if (i + 1 != topk.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << indent.substr(0, indent.size() - 2) << "]";
}

void write_json(
        const Args & args,
        const std::vector<llama_token> & prompt_tokens,
        const LogitsDump & prefill,
        const std::vector<DecodeDump> & decode,
        const std::vector<llama_token> & generated_tokens) {
    std::ofstream out(args.output);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + args.output);
    }

    out << std::setprecision(9);
    out << "{\n";
    out << "  \"model\": \"" << json_escape(args.model) << "\",\n";
    out << "  \"prompt\": {\n";
    out << "    \"text\": \"" << json_escape(args.prompt) << "\",\n";
    out << "    \"tokens\": [";
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << prompt_tokens[i];
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"settings\": {\n";
    out << "    \"n_predict\": " << args.n_predict << ",\n";
    out << "    \"logits_top_k\": " << args.logits_top_k << ",\n";
    out << "    \"sampler\": \"greedy\"\n";
    out << "  },\n";
    out << "  \"prefill\": {\n";
    out << "    \"position\": " << prefill.position << ",\n";
    out << "    \"topk_logits\": ";
    write_topk(out, prefill.topk_logits, "      ");
    out << ",\n";
    out << "    \"greedy_next_token\": " << prefill.greedy_next_token << "\n";
    out << "  },\n";
    out << "  \"decode\": [";
    if (!decode.empty()) {
        out << "\n";
        for (size_t i = 0; i < decode.size(); ++i) {
            const DecodeDump & step = decode[i];
            const LogitsDump & logits = step.logits;
            out << "    {\n";
            out << "      \"step\": " << step.step << ",\n";
            out << "      \"position\": " << logits.position << ",\n";
            out << "      \"topk_logits\": ";
            write_topk(out, logits.topk_logits, "        ");
            out << ",\n";
            out << "      \"greedy_next_token\": " << logits.greedy_next_token << "\n";
            out << "    }";
            if (i + 1 != decode.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "  ";
    }
    out << "],\n";
    out << "  \"generated_tokens\": [";
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << generated_tokens[i];
    }
    out << "]\n";
    out << "}\n";

    if (!out) {
        throw std::runtime_error("failed while writing output file: " + args.output);
    }
}

LogitsDump make_logits_dump(llama_context * ctx, int32_t n_vocab, int top_k, llama_pos position) {
    LogitsDump dump;
    dump.position = position;
    dump.topk_logits = collect_topk_logits(llama_get_logits_ith(ctx, -1), n_vocab, top_k);
    dump.greedy_next_token = greedy_token_from_topk(dump.topk_logits);
    return dump;
}

DecodeDump make_decode_dump(llama_context * ctx, int32_t n_vocab, int top_k, int step, llama_pos position) {
    return { step, make_logits_dump(ctx, n_vocab, top_k, position) };
}

int run(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const Args args = parse_args(argc, argv);

    LlamaBackend backend;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;

    LlamaModel model(args.model, model_params);
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    if (vocab == nullptr) {
        throw std::runtime_error("model has no vocabulary");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        throw std::runtime_error("model vocabulary is empty");
    }

    const std::vector<llama_token> prompt_tokens = tokenize_prompt(vocab, args.prompt);

    const uint32_t n_ctx = static_cast<uint32_t>(prompt_tokens.size() + static_cast<size_t>(args.n_predict));
    const uint32_t n_batch = static_cast<uint32_t>(std::max<size_t>(prompt_tokens.size(), 1));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_batch;
    ctx_params.no_perf = true;

    LlamaContext ctx(model.get(), ctx_params);

    LlamaBatch batch(static_cast<int32_t>(std::max<size_t>(prompt_tokens.size(), 1)));
    batch.clear();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        batch.add(prompt_tokens[i], static_cast<llama_pos>(i), i + 1 == prompt_tokens.size());
    }

    decode_batch(ctx.get(), batch.get());

    const LogitsDump prefill = make_logits_dump(
        ctx.get(),
        n_vocab,
        args.logits_top_k,
        static_cast<llama_pos>(prompt_tokens.size() - 1));

    std::vector<DecodeDump> decode;
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(static_cast<size_t>(args.n_predict));
    decode.reserve(static_cast<size_t>(args.n_predict - 1));

    generated_tokens.push_back(prefill.greedy_next_token);

    for (int i = 0; i < args.n_predict - 1; ++i) {
        const llama_pos pos = static_cast<llama_pos>(prompt_tokens.size() + static_cast<size_t>(i));
        const llama_token input_token = generated_tokens.back();

        batch.clear();
        batch.add(input_token, pos, true);
        decode_batch(ctx.get(), batch.get());

        DecodeDump step = make_decode_dump(ctx.get(), n_vocab, args.logits_top_k, i, pos);
        generated_tokens.push_back(step.logits.greedy_next_token);
        decode.push_back(std::move(step));
    }

    write_json(args, prompt_tokens, prefill, decode, generated_tokens);

    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
