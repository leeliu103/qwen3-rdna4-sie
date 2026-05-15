#include "qwen3_weights.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *err, size_t err_size, const char *fmt, ...) {
    va_list ap;

    if (!err || err_size == 0) return;

    va_start(ap, fmt);
    vsnprintf(err, err_size, fmt, ap);
    va_end(ap);

    err[err_size - 1] = '\0';
}

static bool required_u32(
    const gguf_file *g,
    const char *key,
    uint32_t *out,
    char *err,
    size_t err_size
) {
    if (!gguf_get_u32(g, key, out)) {
        set_error(err, err_size, "required metadata key is missing or not uint32: %s", key);
        return false;
    }
    return true;
}

static bool config_expect_u32(
    const char *name,
    uint32_t got,
    uint32_t expected,
    char *err,
    size_t err_size
) {
    if (got == expected) return true;

    set_error(err, err_size, "expected %s=%u for Qwen3, got %u", name, expected, got);
    return false;
}

static bool validate_config(const gguf_file *g, char *err, size_t err_size) {
    uint32_t n_layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t n_head_dim = 0;
    uint32_t n_value_dim = 0;

    if (!required_u32(g, "qwen3.block_count", &n_layer, err, err_size)) return false;
    if (!required_u32(g, "qwen3.embedding_length", &n_embd, err, err_size)) return false;
    if (!required_u32(g, "qwen3.feed_forward_length", &n_ff, err, err_size)) return false;
    if (!required_u32(g, "qwen3.attention.head_count", &n_head, err, err_size)) return false;
    if (!required_u32(g, "qwen3.attention.head_count_kv", &n_head_kv, err, err_size)) return false;
    if (!required_u32(g, "qwen3.attention.key_length", &n_head_dim, err, err_size)) return false;
    if (!required_u32(g, "qwen3.attention.value_length", &n_value_dim, err, err_size)) return false;

    if (!config_expect_u32("block_count", n_layer, QWEN3_N_LAYER, err, err_size)) return false;
    if (!config_expect_u32("embedding_length", n_embd, QWEN3_N_EMBD, err, err_size)) return false;
    if (!config_expect_u32("feed_forward_length", n_ff, QWEN3_N_FF, err, err_size)) return false;
    if (!config_expect_u32("attention.head_count", n_head, QWEN3_N_HEAD, err, err_size)) return false;
    if (!config_expect_u32("attention.head_count_kv", n_head_kv, QWEN3_N_HEAD_KV, err, err_size)) return false;
    if (!config_expect_u32("attention.key_length", n_head_dim, QWEN3_N_HEAD_DIM, err, err_size)) return false;
    if (!config_expect_u32("attention.value_length", n_value_dim, QWEN3_N_HEAD_DIM, err, err_size)) return false;

    return true;
}

static const gguf_tensor *required_tensor(
    const gguf_file *g,
    const char *name,
    char *err,
    size_t err_size
) {
    const gguf_tensor *t = gguf_find_tensor(g, name);
    if (!t) {
        set_error(err, err_size, "required tensor is missing: %s", name);
        return NULL;
    }
    return t;
}

static const gguf_tensor *required_tensorf(
    const gguf_file *g,
    char *err,
    size_t err_size,
    const char *fmt,
    uint32_t layer
) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);

    if (n < 0 || (size_t)n >= sizeof(name)) {
        set_error(err, err_size, "tensor name is too long");
        return NULL;
    }

    return required_tensor(g, name, err, err_size);
}

bool qwen3_weights_bind(
    qwen3_weights *w,
    const gguf_file *g,
    char *err,
    size_t err_size
) {
    if (!w) {
        set_error(err, err_size, "qwen3_weights_bind called with null weights");
        return false;
    }

    memset(w, 0, sizeof(*w));

    if (!g) {
        set_error(err, err_size, "qwen3_weights_bind called with null gguf_file");
        return false;
    }

    if (!validate_config(g, err, err_size)) return false;

    w->token_embd = required_tensor(g, "token_embd.weight", err, err_size);
    if (!w->token_embd) return false;
    w->output_norm = required_tensor(g, "output_norm.weight", err, err_size);
    if (!w->output_norm) return false;

    w->cls_out = gguf_find_tensor(g, "cls_out.weight");

    w->output = gguf_find_tensor(g, "output.weight");
    if (w->output) {
        w->output_is_tied = false;
    } else {
        w->output = w->token_embd;
        w->output_is_tied = true;
    }

    for (uint32_t il = 0; il < QWEN3_N_LAYER; il++) {
        qwen3_layer_weights *l = &w->layer[il];

        l->attn_norm = required_tensorf(g, err, err_size, "blk.%u.attn_norm.weight", il);
        if (!l->attn_norm) return false;
        l->attn_q = required_tensorf(g, err, err_size, "blk.%u.attn_q.weight", il);
        if (!l->attn_q) return false;
        l->attn_k = required_tensorf(g, err, err_size, "blk.%u.attn_k.weight", il);
        if (!l->attn_k) return false;
        l->attn_v = required_tensorf(g, err, err_size, "blk.%u.attn_v.weight", il);
        if (!l->attn_v) return false;
        l->attn_output = required_tensorf(g, err, err_size, "blk.%u.attn_output.weight", il);
        if (!l->attn_output) return false;
        l->attn_q_norm = required_tensorf(g, err, err_size, "blk.%u.attn_q_norm.weight", il);
        if (!l->attn_q_norm) return false;
        l->attn_k_norm = required_tensorf(g, err, err_size, "blk.%u.attn_k_norm.weight", il);
        if (!l->attn_k_norm) return false;
        l->ffn_norm = required_tensorf(g, err, err_size, "blk.%u.ffn_norm.weight", il);
        if (!l->ffn_norm) return false;
        l->ffn_gate = required_tensorf(g, err, err_size, "blk.%u.ffn_gate.weight", il);
        if (!l->ffn_gate) return false;
        l->ffn_down = required_tensorf(g, err, err_size, "blk.%u.ffn_down.weight", il);
        if (!l->ffn_down) return false;
        l->ffn_up = required_tensorf(g, err, err_size, "blk.%u.ffn_up.weight", il);
        if (!l->ffn_up) return false;
    }

    return true;
}
