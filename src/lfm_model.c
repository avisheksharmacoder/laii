#define _POSIX_C_SOURCE 200809L
#include "lfm_model.h"
#include "lfm_defaults.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const void* find_tensor_data(const GGUFFile *file, const char *name, GGMLType *out_type) {
    const GGUFTensor *t = gguf_get_tensor(file, name);
    if (t) {
        if (out_type) *out_type = t->type;
        return t->data;
    }
    return NULL;
}

bool lfm_model_load(LFMModel *model, const char *filepath) {
    if (!model || !filepath) return false;
    memset(model, 0, sizeof(LFMModel));

    if (!gguf_open(&model->gguf, filepath)) {
        fprintf(stderr, "[Model] Failed to open GGUF model: %s\n", filepath);
        return false;
    }

    const GGUFFile *file = &model->gguf;

    // 1. Extract Hyperparameters
    const GGUFValue *arch_val = gguf_get_val(file, "general.architecture");
    if (arch_val && arch_val->type == GGUF_TYPE_STRING && arch_val->val_str.data) {
        strncpy(model->config.architecture, arch_val->val_str.data, sizeof(model->config.architecture) - 1);
    } else {
        strncpy(model->config.architecture, "lfm2", sizeof(model->config.architecture) - 1);
    }

    model->config.vocab_size = gguf_get_u32(file, "lfm2.vocab_size", 0);
    if (model->config.vocab_size == 0) {
        model->config.vocab_size = gguf_get_u32(file, "general.vocab_size", 65536);
    }

    model->config.hidden_dim = gguf_get_u32(file, "lfm2.embedding_length", 0);
    if (model->config.hidden_dim == 0) {
        model->config.hidden_dim = gguf_get_u32(file, "general.embedding_length", 2048);
    }

    model->config.ffn_dim = gguf_get_u32(file, "lfm2.feed_forward_length", 0);
    if (model->config.ffn_dim == 0) {
        model->config.ffn_dim = gguf_get_u32(file, "general.feed_forward_length", 8192);
    }

    model->config.n_layers = gguf_get_u32(file, "lfm2.block_count", 0);
    if (model->config.n_layers == 0) {
        model->config.n_layers = gguf_get_u32(file, "general.block_count", 16);
    }
    if (model->config.n_layers > LFM_MAX_LAYERS) {
        model->config.n_layers = LFM_MAX_LAYERS;
    }

    // Query heads (scalar or array)
    const GGUFValue *hq_val = gguf_get_val(file, "lfm2.attention.head_count");
    if (!hq_val) hq_val = gguf_get_val(file, "attention.head_count");
    if (hq_val && hq_val->type == GGUF_TYPE_ARRAY && hq_val->val_arr.len > 0) {
        uint32_t *arr = (uint32_t*)hq_val->val_arr.data;
        model->config.n_heads_q = 0;
        for (uint64_t i = 0; i < hq_val->val_arr.len; ++i) {
            if (arr[i] > model->config.n_heads_q) model->config.n_heads_q = arr[i];
        }
    } else {
        model->config.n_heads_q = gguf_get_u32(file, "lfm2.attention.head_count", 16);
    }

    // KV heads array routing (lfm2.attention.head_count_kv: [0, 0, 8, 0, 0, 8, ...])
    bool routing_determined = false;
    const GGUFValue *hkv_val = gguf_get_val(file, "lfm2.attention.head_count_kv");
    if (!hkv_val) hkv_val = gguf_get_val(file, "attention.head_count_kv");

    if (hkv_val && hkv_val->type == GGUF_TYPE_ARRAY && hkv_val->val_arr.len > 0) {
        uint32_t *arr = (uint32_t*)hkv_val->val_arr.data;
        model->config.n_heads_kv = 0;
        for (uint32_t i = 0; i < model->config.n_layers && i < hkv_val->val_arr.len; ++i) {
            model->config.layer_heads_kv[i] = arr[i];
            if (arr[i] == 0) {
                model->config.is_conv_layer[i] = true;
            } else {
                model->config.is_conv_layer[i] = false;
                if (arr[i] > model->config.n_heads_kv) {
                    model->config.n_heads_kv = arr[i];
                }
            }
        }
        routing_determined = true;
    } else {
        model->config.n_heads_kv = gguf_get_u32(file, "lfm2.attention.head_count_kv", 4);
        for (uint32_t i = 0; i < model->config.n_layers; ++i) {
            model->config.layer_heads_kv[i] = model->config.n_heads_kv;
        }
    }
    if (model->config.n_heads_kv == 0) model->config.n_heads_kv = 4;
    model->config.head_dim = model->config.n_heads_q > 0 ? (model->config.hidden_dim / model->config.n_heads_q) : 64;

    model->config.conv_L_cache = gguf_get_u32(file, "lfm2.shortconv.l_cache", 0);
    if (model->config.conv_L_cache == 0) {
        model->config.conv_L_cache = gguf_get_u32(file, "lfm2.shortconv_l_cache", 3);
    }
    model->config.max_seq_len = gguf_get_u32(file, "lfm2.context_length", (uint32_t)LFM_DEFAULT_CTX_LEN);
    model->config.norm_eps = gguf_get_f32(file, "lfm2.attention.layer_norm_rms_epsilon", 1e-5f);
    model->config.rope_theta = gguf_get_f32(file, "lfm2.rope.freq_base", 1000000.0f);

    // If routing wasn't determined by head_count_kv array, check layer_types or shortconv tensor
    if (!routing_determined) {
        const GGUFValue *ltypes_val = gguf_get_val(file, "lfm2.layer_types");
        if (ltypes_val && ltypes_val->type == GGUF_TYPE_ARRAY && ltypes_val->val_arr.type == GGUF_TYPE_STRING) {
            GGUFString *ltypes = (GGUFString*)ltypes_val->val_arr.data;
            for (uint32_t i = 0; i < model->config.n_layers && i < ltypes_val->val_arr.len; ++i) {
                if (ltypes[i].data && strcmp(ltypes[i].data, "full_attention") == 0) {
                    model->config.is_conv_layer[i] = false;
                } else {
                    model->config.is_conv_layer[i] = true;
                }
            }
        } else {
            // Fallback: Check if blk.i.shortconv.conv.weight exists for each layer
            for (uint32_t i = 0; i < model->config.n_layers; ++i) {
                char tname[128];
                snprintf(tname, sizeof(tname), "blk.%u.shortconv.conv.weight", i);
                model->config.is_conv_layer[i] = (gguf_get_tensor(file, tname) != NULL);
            }
        }
    }

    // 2. Load Global Tensors
    model->tok_embeddings = find_tensor_data(file, "token_embd.weight", &model->tok_embeddings_type);
    
    // LFM2 uses token_embd_norm.weight for final RMSNorm; fall back to output_norm / result_norm
    model->output_norm = (const float*)find_tensor_data(file, "token_embd_norm.weight", NULL);
    if (!model->output_norm) {
        model->output_norm = (const float*)find_tensor_data(file, "output_norm.weight", NULL);
    }
    if (!model->output_norm) {
        model->output_norm = (const float*)find_tensor_data(file, "result_norm.weight", NULL);
    }

    // LFM2 often ties output.weight directly to token_embd.weight
    model->lm_head = find_tensor_data(file, "output.weight", &model->lm_head_type);
    if (!model->lm_head) {
        model->lm_head = model->tok_embeddings;
        model->lm_head_type = model->tok_embeddings_type;
    }

    // 3. Load Per-Layer Tensors
    for (uint32_t i = 0; i < model->config.n_layers; ++i) {
        LFMLayerWeights *lw = &model->layers[i];
        char tname[128];

        // Norms
        snprintf(tname, sizeof(tname), "blk.%u.attn_norm.weight", i);
        lw->operator_norm = (const float*)find_tensor_data(file, tname, NULL);
        if (!lw->operator_norm) {
            snprintf(tname, sizeof(tname), "blk.%u.operator_norm.weight", i);
            lw->operator_norm = (const float*)find_tensor_data(file, tname, NULL);
        }

        snprintf(tname, sizeof(tname), "blk.%u.ffn_norm.weight", i);
        lw->ffn_norm = (const float*)find_tensor_data(file, tname, NULL);

        if (model->config.is_conv_layer[i]) {
            // ShortConv Layer
            snprintf(tname, sizeof(tname), "blk.%u.shortconv.in_proj.weight", i);
            lw->shortconv_in_proj = find_tensor_data(file, tname, &lw->shortconv_in_proj_type);

            snprintf(tname, sizeof(tname), "blk.%u.shortconv.conv.weight", i);
            lw->shortconv_conv = (const float*)find_tensor_data(file, tname, NULL);

            snprintf(tname, sizeof(tname), "blk.%u.shortconv.conv.bias", i);
            lw->shortconv_conv_bias = (const float*)find_tensor_data(file, tname, NULL);

            snprintf(tname, sizeof(tname), "blk.%u.shortconv.out_proj.weight", i);
            lw->shortconv_out_proj = find_tensor_data(file, tname, &lw->shortconv_out_proj_type);
        } else {
            // Attention Layer
            snprintf(tname, sizeof(tname), "blk.%u.attn_q.weight", i);
            lw->q_proj = find_tensor_data(file, tname, &lw->q_proj_type);

            snprintf(tname, sizeof(tname), "blk.%u.attn_k.weight", i);
            lw->k_proj = find_tensor_data(file, tname, &lw->k_proj_type);

            snprintf(tname, sizeof(tname), "blk.%u.attn_v.weight", i);
            lw->v_proj = find_tensor_data(file, tname, &lw->v_proj_type);

            snprintf(tname, sizeof(tname), "blk.%u.attn_q_norm.weight", i);
            lw->q_norm = (const float*)find_tensor_data(file, tname, NULL);

            snprintf(tname, sizeof(tname), "blk.%u.attn_k_norm.weight", i);
            lw->k_norm = (const float*)find_tensor_data(file, tname, NULL);

            snprintf(tname, sizeof(tname), "blk.%u.attn_output.weight", i);
            lw->attn_out_proj = find_tensor_data(file, tname, &lw->attn_out_proj_type);
            if (!lw->attn_out_proj) {
                snprintf(tname, sizeof(tname), "blk.%u.attn_out.weight", i);
                lw->attn_out_proj = find_tensor_data(file, tname, &lw->attn_out_proj_type);
            }
        }

        // FFN (SwiGLU)
        snprintf(tname, sizeof(tname), "blk.%u.ffn_gate.weight", i);
        lw->ffn_gate = find_tensor_data(file, tname, &lw->ffn_gate_type);

        snprintf(tname, sizeof(tname), "blk.%u.ffn_up.weight", i);
        lw->ffn_up = find_tensor_data(file, tname, &lw->ffn_up_type);

        snprintf(tname, sizeof(tname), "blk.%u.ffn_down.weight", i);
        lw->ffn_down = find_tensor_data(file, tname, &lw->ffn_down_type);
    }

    return true;
}

void lfm_model_free(LFMModel *model) {
    if (model) {
        gguf_close(&model->gguf);
    }
}
