#ifndef QWEN3_WEIGHTS_H
#define QWEN3_WEIGHTS_H

#include <stdbool.h>

#include "../../core/gguf_loader.h"
#include "qwen3_config.h"

typedef struct {
    const gguf_tensor *attn_norm;

    const gguf_tensor *attn_q;
    const gguf_tensor *attn_k;
    const gguf_tensor *attn_v;
    const gguf_tensor *attn_output;

    const gguf_tensor *attn_q_norm;
    const gguf_tensor *attn_k_norm;

    const gguf_tensor *ffn_norm;
    const gguf_tensor *ffn_gate;
    const gguf_tensor *ffn_down;
    const gguf_tensor *ffn_up;
} qwen3_layer_weights;

typedef struct {
    const gguf_tensor *token_embd;
    const gguf_tensor *output_norm;

    const gguf_tensor *output;
    bool output_is_tied;

    const gguf_tensor *cls_out;

    qwen3_layer_weights layer[QWEN3_N_LAYER];
} qwen3_weights;

#endif
