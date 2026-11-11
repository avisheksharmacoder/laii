#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef LFM2_LOADER_INCLUDED
#define LFM2_LOADER_INCLUDED 1

#define GGUF_MAGIC 0x46554747U /* "GGUF" in little-endian */
#define GGUF_DEFAULT_ALIGNMENT 32
#define LFM2_MAX_LAYERS 64
#define LFM2_MAX_DIMS 4

/* ============================================================================
 * 1. GGML Quantization Types & Block Geometry Table
 * ============================================================================ */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT
} ggml_type_t;

typedef struct {
    const char *name;
    uint32_t block_elements; /* Number of weights per quantization block */
    uint32_t block_bytes;    /* Size in bytes of one quantization block  */
} ggml_type_traits_t;

static bool get_ggml_type_traits(uint32_t type, ggml_type_traits_t *out) {
    switch (type) {
        case GGML_TYPE_F32:    *out = (ggml_type_traits_t){"F32",    1,   4};   return true;
        case GGML_TYPE_F16:    *out = (ggml_type_traits_t){"F16",    1,   2};   return true;
        case GGML_TYPE_BF16:   *out = (ggml_type_traits_t){"BF16",   1,   2};   return true;
        case GGML_TYPE_Q4_0:   *out = (ggml_type_traits_t){"Q4_0",   32,  18};  return true;
        case GGML_TYPE_Q4_1:   *out = (ggml_type_traits_t){"Q4_1",   32,  20};  return true;
        case GGML_TYPE_Q5_0:   *out = (ggml_type_traits_t){"Q5_0",   32,  22};  return true;
        case GGML_TYPE_Q5_1:   *out = (ggml_type_traits_t){"Q5_1",   32,  24};  return true;
        case GGML_TYPE_Q8_0:   *out = (ggml_type_traits_t){"Q8_0",   32,  34};  return true;
        case GGML_TYPE_Q8_1:   *out = (ggml_type_traits_t){"Q8_1",   32,  36};  return true;
        case GGML_TYPE_Q2_K:   *out = (ggml_type_traits_t){"Q2_K",   256, 84};  return true;
        case GGML_TYPE_Q3_K:   *out = (ggml_type_traits_t){"Q3_K",   256, 110}; return true;
        case GGML_TYPE_Q4_K:   *out = (ggml_type_traits_t){"Q4_K",   256, 144}; return true;
        case GGML_TYPE_Q5_K:   *out = (ggml_type_traits_t){"Q5_K",   256, 176}; return true;
        case GGML_TYPE_Q6_K:   *out = (ggml_type_traits_t){"Q6_K",   256, 210}; return true;
        case GGML_TYPE_Q8_K:   *out = (ggml_type_traits_t){"Q8_K",   256, 292}; return true;
        case GGML_TYPE_IQ4_NL: *out = (ggml_type_traits_t){"IQ4_NL", 32,  18};  return true;
        case GGML_TYPE_IQ4_XS: *out = (ggml_type_traits_t){"IQ4_XS", 256, 136}; return true;
        default:               *out = (ggml_type_traits_t){"UNK",    0,   0};   return false;
    }
}

/* ============================================================================
 * 2. GGUF v3 Metadata Value Types
 * ============================================================================ */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
} gguf_val_type_t;

/* ============================================================================
 * 3. LFM2 / LFM2.5 Tensor & Architecture Structures
 * ============================================================================ */
typedef struct {
    char     name[96];
    uint32_t n_dims;
    uint64_t ne[LFM2_MAX_DIMS]; /* ne[0] is innermost contiguous dimension */
    uint32_t type;              /* ggml_type_t */
    uint64_t rel_offset;        /* Offset relative to start of data section */
    size_t   bytes;             /* Computed byte size of tensor payload */
    const void *data;           /* Direct zero-copy pointer into mmap buffer */
} GGUFTensor;

#ifndef LFM2_BLOCK_TYPE_DEFINED
#define LFM2_BLOCK_TYPE_DEFINED 1
typedef enum {
    LFM2_BLOCK_SHORTCONV = 0,   /* Double-Gated LIV Short Convolution Block */
    LFM2_BLOCK_GQA_ATTN  = 1,   /* Global Grouped-Query Attention Block     */
    LFM2_BLOCK_SWA_ATTN  = 2    /* Sliding-Window Grouped-Query Attention   */
} lfm2_block_type_t;
#endif

static inline bool lfm2_block_is_attn(lfm2_block_type_t bt) {
    return bt == LFM2_BLOCK_GQA_ATTN || bt == LFM2_BLOCK_SWA_ATTN;
}

typedef struct {
    uint32_t          layer_idx;
    lfm2_block_type_t block_type;

    /* Shared Layer Pre-Norms, Post-Norms (LFM3 Sandwich Norm) & SwiGLU FFN Tensors */
    const GGUFTensor *attn_norm;      /* blk.i.attn_norm.weight [n_embd] (F32)            */
    const GGUFTensor *post_attn_norm; /* blk.i.post_attention_norm.weight [n_embd] (LFM3) */
    const GGUFTensor *ffn_norm;       /* blk.i.ffn_norm.weight  [n_embd] (F32)            */
    const GGUFTensor *post_ffn_norm;  /* blk.i.post_ffw_norm.weight [n_embd] (LFM3)       */
    const GGUFTensor *ffn_gate;       /* blk.i.ffn_gate.weight  [n_embd, n_ff]            */
    const GGUFTensor *ffn_up;         /* blk.i.ffn_up.weight    [n_embd, n_ff]            */
    const GGUFTensor *ffn_down;       /* blk.i.ffn_down.weight  [n_ff, n_embd]            */

    /* Operator Union: Either LIV ShortConv or GQA/SWA Attention */
    union {
        struct {
            const GGUFTensor *in_proj;  /* blk.i.shortconv.in_proj.weight  [n_embd, 3*n_embd] */
            const GGUFTensor *conv;     /* blk.i.shortconv.conv.weight     [l_cache, n_embd]  */
            const GGUFTensor *out_proj; /* blk.i.shortconv.out_proj.weight [n_embd, n_embd]   */
        } shortconv;

        struct {
            const GGUFTensor *wq;       /* blk.i.attn_q.weight      [n_embd, n_embd]    */
            const GGUFTensor *wk;       /* blk.i.attn_k.weight      [n_embd, n_embd_kv] */
            const GGUFTensor *wv;       /* blk.i.attn_v.weight      [n_embd, n_embd_kv] */
            const GGUFTensor *q_norm;   /* blk.i.attn_q_norm.weight [head_dim] (F32)    */
            const GGUFTensor *k_norm;   /* blk.i.attn_k_norm.weight [head_dim] (F32)    */
            const GGUFTensor *gate;     /* blk.i.attn_gate.weight   [n_embd, n_head_q]  */
            const GGUFTensor *wo;       /* blk.i.attn_output.weight [n_embd, n_embd]    */
        } attn;
    };
} LFM2Layer;

typedef struct {
    char     arch[32];
    bool     is_lfm3;
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t n_ctx;
    uint32_t n_head_q;
    uint32_t head_dim;                   /* Explicit key/value length (64 LFM2, 128 LFM3) */
    uint32_t rope_dim;                   /* RoPE dim count (typically == head_dim)        */
    uint32_t n_head_kv[LFM2_MAX_LAYERS]; /* 0 = shortconv layer, >0 = GQA/SWA layer       */
    bool     is_swa_layer[LFM2_MAX_LAYERS]; /* true = sliding_window attention (LFM3)     */
    uint32_t sliding_window;             /* e.g. 256 for LFM3 SWA layers, 0 for LFM2      */
    uint32_t shortconv_l_cache;          /* 3 for LFM2, 7 for LFM3                        */
    float    rms_norm_eps;
    float    rope_freq_base;
    float    rope_freq_base_swa;         /* e.g. 10000.0f for LFM3 sliding_attention      */
    uint32_t alignment;
    uint32_t n_experts;
    uint32_t n_experts_used;
    int8_t   add_bos_token;              /* -1 unset, 0 false, 1 true                     */
} LFM2Config;

typedef struct {
    int         fd;
    size_t      file_size;
    const uint8_t *mmap_base;
    const uint8_t *tensor_data_base;

    LFM2Config  cfg;

    uint64_t    n_tensors;
    GGUFTensor *tensors;

    /* Global Model Tensors */
    const GGUFTensor *token_embd;      /* token_embd.weight */
    const GGUFTensor *output_norm;     /* token_embd_norm.weight or output_norm.weight */
    const GGUFTensor *output;          /* output.weight (falls back to token_embd) */
    const GGUFTensor *rope_freqs;      /* rope_freqs.weight [rope_dim/2] (LFM3 partial RoPE) */

    /* Hybrid Layer Stack */
    LFM2Layer   layers[LFM2_MAX_LAYERS];
} LFM2Model;

/* ============================================================================
 * 4. Safe Binary Cursor Reader for mmap Buffer
 * ============================================================================ */
typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
} ByteCursor;

static bool cursor_read_bytes(ByteCursor *c, void *dst, size_t len) {
    if (c->pos + len > c->size) return false;
    memcpy(dst, c->base + c->pos, len);
    c->pos += len;
    return true;
}

static bool cursor_skip(ByteCursor *c, size_t len) {
    if (c->pos + len > c->size) return false;
    c->pos += len;
    return true;
}

static bool cursor_read_u32(ByteCursor *c, uint32_t *out) {
    return cursor_read_bytes(c, out, sizeof(uint32_t));
}

static bool cursor_read_u64(ByteCursor *c, uint64_t *out) {
    return cursor_read_bytes(c, out, sizeof(uint64_t));
}

static bool cursor_read_f32(ByteCursor *c, float *out) {
    return cursor_read_bytes(c, out, sizeof(float));
}

/* GGUF strings are stored as uint64_t length followed by non-null-terminated bytes */
static bool cursor_read_gguf_str(ByteCursor *c, char *buf, size_t buf_cap, uint64_t *raw_len_out) {
    uint64_t len = 0;
    if (!cursor_read_u64(c, &len)) return false;
    if (c->pos + len > c->size) return false;

    if (raw_len_out) *raw_len_out = len;
    if (buf && buf_cap > 0) {
        size_t copy_len = (len < buf_cap - 1) ? (size_t)len : (buf_cap - 1);
        memcpy(buf, c->base + c->pos, copy_len);
        buf[copy_len] = '\0';
    }
    c->pos += (size_t)len;
    return true;
}

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

/* Recursively skips any uninterested GGUF metadata value (including arrays of strings) */
static bool cursor_skip_gguf_value(ByteCursor *c, uint32_t val_type) {
    size_t s_size = gguf_scalar_size(val_type);
    if (s_size > 0) {
        return cursor_skip(c, s_size);
    }
    if (val_type == GGUF_TYPE_STRING) {
        return cursor_read_gguf_str(c, NULL, 0, NULL);
    }
    if (val_type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!cursor_read_u32(c, &elem_type) || !cursor_read_u64(c, &count)) {
            return false;
        }
        size_t elem_scalar = gguf_scalar_size(elem_type);
        if (elem_scalar > 0) {
            return cursor_skip(c, (size_t)count * elem_scalar);
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (!cursor_skip_gguf_value(c, elem_type)) return false;
        }
        return true;
    }
    return false;
}

static bool read_gguf_u32_flexible(ByteCursor *c, uint32_t val_type, uint32_t *out) {
    if (val_type == GGUF_TYPE_UINT32 || val_type == GGUF_TYPE_INT32) {
        return cursor_read_u32(c, out);
    }
    if (val_type == GGUF_TYPE_UINT64 || val_type == GGUF_TYPE_INT64) {
        uint64_t tmp = 0;
        if (!cursor_read_u64(c, &tmp)) return false;
        *out = (uint32_t)tmp;
        return true;
    }
    return false;
}

static bool lfm2_key_has_suffix(const char *key, const char *suffix) {
    size_t klen = strlen(key);
    size_t slen = strlen(suffix);
    return (klen >= slen) && (strcmp(key + (klen - slen), suffix) == 0);
}

/* ============================================================================
 * 5. Metadata Extractor for LFM2 / LFM2.5 / LFM3 Hyperparameters
 * ============================================================================ */
static bool parse_lfm2_metadata_kv(ByteCursor *c, LFM2Config *cfg) {
    char key[128];
    uint32_t val_type = 0;

    if (!cursor_read_gguf_str(c, key, sizeof(key), NULL)) return false;
    if (!cursor_read_u32(c, &val_type)) return false;

    if (strcmp(key, "general.architecture") == 0 && val_type == GGUF_TYPE_STRING) {
        if (!cursor_read_gguf_str(c, cfg->arch, sizeof(cfg->arch), NULL)) return false;
        cfg->is_lfm3 = (strcmp(cfg->arch, "lfm3") == 0);
        return true;
    }
    if (strcmp(key, "general.alignment") == 0) {
        return read_gguf_u32_flexible(c, val_type, &cfg->alignment);
    }
    if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0 && val_type == GGUF_TYPE_BOOL) {
        uint8_t b = 0;
        if (!cursor_read_bytes(c, &b, 1)) return false;
        cfg->add_bos_token = b ? 1 : 0;
        return true;
    }
    if (lfm2_key_has_suffix(key, ".block_count")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_layers);
    }
    if (lfm2_key_has_suffix(key, ".embedding_length")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_embd);
    }
    if (lfm2_key_has_suffix(key, ".feed_forward_length") && val_type != GGUF_TYPE_ARRAY) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_ff);
    }
    if (lfm2_key_has_suffix(key, ".context_length")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_ctx);
    }
    if (lfm2_key_has_suffix(key, ".vocab_size")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_vocab);
    }
    if (lfm2_key_has_suffix(key, ".shortconv.l_cache")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->shortconv_l_cache);
    }
    if (lfm2_key_has_suffix(key, ".expert_count")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_experts);
    }
    if (lfm2_key_has_suffix(key, ".expert_used_count")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->n_experts_used);
    }
    if (lfm2_key_has_suffix(key, ".attention.layer_norm_rms_epsilon") && val_type == GGUF_TYPE_FLOAT32) {
        return cursor_read_f32(c, &cfg->rms_norm_eps);
    }
    if (lfm2_key_has_suffix(key, ".attention.key_length")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->head_dim);
    }
    if (lfm2_key_has_suffix(key, ".attention.sliding_window")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->sliding_window);
    }
    if (lfm2_key_has_suffix(key, ".attention.sliding_window_pattern") && val_type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!cursor_read_u32(c, &elem_type) || !cursor_read_u64(c, &count)) return false;
        for (uint64_t i = 0; i < count; ++i) {
            uint8_t b = 0;
            if (elem_type == GGUF_TYPE_BOOL) {
                if (!cursor_read_bytes(c, &b, 1)) return false;
            } else {
                uint32_t v = 0;
                if (!read_gguf_u32_flexible(c, elem_type, &v)) return false;
                b = (v != 0);
            }
            if (i < LFM2_MAX_LAYERS) cfg->is_swa_layer[i] = (b != 0);
        }
        return true;
    }
    if (lfm2_key_has_suffix(key, ".rope.dimension_count")) {
        return read_gguf_u32_flexible(c, val_type, &cfg->rope_dim);
    }
    if (lfm2_key_has_suffix(key, ".rope.freq_base_swa") && val_type == GGUF_TYPE_FLOAT32) {
        return cursor_read_f32(c, &cfg->rope_freq_base_swa);
    }
    if (lfm2_key_has_suffix(key, ".rope.freq_base") && val_type == GGUF_TYPE_FLOAT32) {
        return cursor_read_f32(c, &cfg->rope_freq_base);
    }

    /* lfm2/lfm3.attention.head_count can be a scalar uint32 or an array */
    if (lfm2_key_has_suffix(key, ".attention.head_count")) {
        if (val_type == GGUF_TYPE_UINT32) {
            return cursor_read_u32(c, &cfg->n_head_q);
        } else if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!cursor_read_u32(c, &elem_type) || !cursor_read_u64(c, &count)) return false;
            for (uint64_t i = 0; i < count; ++i) {
                uint32_t v = 0;
                if (!read_gguf_u32_flexible(c, elem_type, &v)) return false;
                if (v > cfg->n_head_q) cfg->n_head_q = v;
            }
            return true;
        }
    }

    /* lfm2/lfm3.attention.head_count_kv is an array [0, 0, 8, ...] defining hybrid blocks */
    if (lfm2_key_has_suffix(key, ".attention.head_count_kv")) {
        if (val_type == GGUF_TYPE_UINT32) {
            uint32_t v = 0;
            if (!cursor_read_u32(c, &v)) return false;
            for (int i = 0; i < LFM2_MAX_LAYERS; ++i) cfg->n_head_kv[i] = v;
            return true;
        } else if (val_type == GGUF_TYPE_ARRAY) {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!cursor_read_u32(c, &elem_type) || !cursor_read_u64(c, &count)) return false;
            for (uint64_t i = 0; i < count; ++i) {
                uint32_t v = 0;
                if (!read_gguf_u32_flexible(c, elem_type, &v)) return false;
                if (i < LFM2_MAX_LAYERS) cfg->n_head_kv[i] = v;
            }
            return true;
        }
    }

    /* Fallback vocab_size detection from tokenizer.ggml.tokens array length */
    if (strcmp(key, "tokenizer.ggml.tokens") == 0 && val_type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t count = 0;
        if (!cursor_read_u32(c, &elem_type) || !cursor_read_u64(c, &count)) return false;
        if (cfg->n_vocab == 0) cfg->n_vocab = (uint32_t)count;
        for (uint64_t i = 0; i < count; ++i) {
            if (!cursor_skip_gguf_value(c, elem_type)) return false;
        }
        return true;
    }

    return cursor_skip_gguf_value(c, val_type);
}

/* ============================================================================
 * 6. Tensor Lookup & Hybrid Layer Pointer Binding
 * ============================================================================ */
static const GGUFTensor *find_tensor(const LFM2Model *m, const char *name) {
    for (uint64_t i = 0; i < m->n_tensors; ++i) {
        if (strcmp(m->tensors[i].name, name) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static const GGUFTensor *find_layer_tensor(const LFM2Model *m, uint32_t l, const char *suffix) {
    char buf[96];
    snprintf(buf, sizeof(buf), "blk.%u.%s", l, suffix);
    return find_tensor(m, buf);
}

static bool bind_lfm3_tensors(LFM2Model *m) {
    m->cfg.is_lfm3 = true;
    if (m->cfg.head_dim == 0 && m->cfg.n_head_q > 0) {
        m->cfg.head_dim = m->cfg.n_embd / m->cfg.n_head_q;
    }
    if (m->cfg.rope_dim == 0) {
        m->cfg.rope_dim = m->cfg.head_dim;
    }

    m->token_embd = find_tensor(m, "token_embd.weight");
    m->output_norm = find_tensor(m, "token_embd_norm.weight");
    if (!m->output_norm) {
        m->output_norm = find_tensor(m, "output_norm.weight");
    }
    m->output = find_tensor(m, "output.weight");
    if (!m->output) {
        m->output = m->token_embd;
    }
    m->rope_freqs = find_tensor(m, "rope_freqs.weight");

    if (!m->token_embd || !m->output_norm) {
        fprintf(stderr, "[ERROR] LFM3 missing global token_embd or output_norm tensor.\n");
        return false;
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        LFM2Layer *layer = &m->layers[l];
        layer->layer_idx      = l;
        layer->attn_norm      = find_layer_tensor(m, l, "attn_norm.weight");
        layer->post_attn_norm = find_layer_tensor(m, l, "post_attention_norm.weight");
        layer->ffn_norm       = find_layer_tensor(m, l, "ffn_norm.weight");
        layer->post_ffn_norm  = find_layer_tensor(m, l, "post_ffw_norm.weight");
        layer->ffn_gate       = find_layer_tensor(m, l, "ffn_gate.weight");
        layer->ffn_up         = find_layer_tensor(m, l, "ffn_up.weight");
        layer->ffn_down       = find_layer_tensor(m, l, "ffn_down.weight");

        const GGUFTensor *sc_conv = find_layer_tensor(m, l, "shortconv.conv.weight");
        if (sc_conv != NULL || m->cfg.n_head_kv[l] == 0) {
            layer->block_type         = LFM2_BLOCK_SHORTCONV;
            layer->shortconv.in_proj  = find_layer_tensor(m, l, "shortconv.in_proj.weight");
            layer->shortconv.conv     = sc_conv;
            layer->shortconv.out_proj = find_layer_tensor(m, l, "shortconv.out_proj.weight");

            if (!layer->shortconv.in_proj || !layer->shortconv.conv || !layer->shortconv.out_proj) {
                fprintf(stderr, "[ERROR] LFM3 layer %u missing shortconv tensors.\n", l);
                return false;
            }
        } else {
            layer->block_type  = m->cfg.is_swa_layer[l] ? LFM2_BLOCK_SWA_ATTN : LFM2_BLOCK_GQA_ATTN;
            layer->attn.wq     = find_layer_tensor(m, l, "attn_q.weight");
            layer->attn.wk     = find_layer_tensor(m, l, "attn_k.weight");
            layer->attn.wv     = find_layer_tensor(m, l, "attn_v.weight");
            layer->attn.q_norm = find_layer_tensor(m, l, "attn_q_norm.weight");
            layer->attn.k_norm = find_layer_tensor(m, l, "attn_k_norm.weight");
            layer->attn.gate   = find_layer_tensor(m, l, "attn_gate.weight");
            layer->attn.wo     = find_layer_tensor(m, l, "attn_output.weight");

            if (!layer->attn.wq || !layer->attn.wk || !layer->attn.wv || !layer->attn.wo) {
                fprintf(stderr, "[ERROR] LFM3 layer %u missing attention tensors.\n", l);
                return false;
            }
        }
    }
    return true;
}

static bool bind_lfm2_tensors(LFM2Model *m) {
    if (strcmp(m->cfg.arch, "lfm3") == 0) {
        return bind_lfm3_tensors(m);
    }
    if (m->cfg.head_dim == 0 && m->cfg.n_head_q > 0) {
        m->cfg.head_dim = m->cfg.n_embd / m->cfg.n_head_q;
    }
    if (m->cfg.rope_dim == 0) {
        m->cfg.rope_dim = m->cfg.head_dim;
    }

    m->token_embd = find_tensor(m, "token_embd.weight");
    /* LFM2 uses token_embd_norm.weight for final RMSNorm; check both names */
    m->output_norm = find_tensor(m, "token_embd_norm.weight");
    if (!m->output_norm) {
        m->output_norm = find_tensor(m, "output_norm.weight");
    }
    /* LFM2 often ties output.weight directly to token_embd.weight */
    m->output = find_tensor(m, "output.weight");
    if (!m->output) {
        m->output = m->token_embd;
    }

    if (!m->token_embd || !m->output_norm) {
        fprintf(stderr, "[ERROR] Missing global token_embd or token_embd_norm tensor.\n");
        return false;
    }

    for (uint32_t l = 0; l < m->cfg.n_layers; ++l) {
        LFM2Layer *layer = &m->layers[l];
        layer->layer_idx = l;
        layer->attn_norm = find_layer_tensor(m, l, "attn_norm.weight");
        layer->ffn_norm  = find_layer_tensor(m, l, "ffn_norm.weight");
        layer->ffn_gate  = find_layer_tensor(m, l, "ffn_gate.weight");
        layer->ffn_up    = find_layer_tensor(m, l, "ffn_up.weight");
        layer->ffn_down  = find_layer_tensor(m, l, "ffn_down.weight");

        /* Detect ShortConv vs GQA Attention via tensor presence & metadata */
        const GGUFTensor *sc_conv = find_layer_tensor(m, l, "shortconv.conv.weight");
        if (sc_conv != NULL || m->cfg.n_head_kv[l] == 0) {
            layer->block_type         = LFM2_BLOCK_SHORTCONV;
            layer->shortconv.in_proj  = find_layer_tensor(m, l, "shortconv.in_proj.weight");
            layer->shortconv.conv     = sc_conv;
            layer->shortconv.out_proj = find_layer_tensor(m, l, "shortconv.out_proj.weight");

            if (!layer->shortconv.in_proj || !layer->shortconv.conv || !layer->shortconv.out_proj) {
                fprintf(stderr, "[ERROR] Layer %u missing shortconv tensors.\n", l);
                return false;
            }
        } else {
            layer->block_type  = LFM2_BLOCK_GQA_ATTN;
            layer->attn.wq     = find_layer_tensor(m, l, "attn_q.weight");
            layer->attn.wk     = find_layer_tensor(m, l, "attn_k.weight");
            layer->attn.wv     = find_layer_tensor(m, l, "attn_v.weight");
            layer->attn.q_norm = find_layer_tensor(m, l, "attn_q_norm.weight");
            layer->attn.k_norm = find_layer_tensor(m, l, "attn_k_norm.weight");
            layer->attn.wo     = find_layer_tensor(m, l, "attn_output.weight");

            if (!layer->attn.wq || !layer->attn.wk || !layer->attn.wv || !layer->attn.wo) {
                fprintf(stderr, "[ERROR] Layer %u missing GQA attention tensors.\n", l);
                return false;
            }
        }
    }
    return true;
}

/* ============================================================================
 * 7. Main GGUF v3 mmap Loader & Unloader
 * ============================================================================ */
bool lfm2_load_gguf_from_fd(int fd, size_t file_size, LFM2Model *m) {
    if (fd < 0) return false;
    memset(m, 0, sizeof(*m));
    m->fd = fd;
    m->cfg.alignment         = GGUF_DEFAULT_ALIGNMENT;
    m->cfg.shortconv_l_cache = 3;
    m->cfg.rms_norm_eps      = 1e-5f;
    m->cfg.rope_freq_base    = 1e6f;
    m->cfg.rope_freq_base_swa = 10000.0f;
    m->cfg.add_bos_token     = -1;

    if (file_size < 24) {
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 24) {
            perror("fstat");
            return false;
        }
        file_size = (size_t)st.st_size;
    }
    m->file_size = file_size;

    void *mapped = mmap(NULL, m->file_size, PROT_READ, MAP_SHARED, m->fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return false;
    }
    m->mmap_base = (const uint8_t *)mapped;

    /* Hint Linux kernel for contiguous sequential read-ahead & transparent huge pages */
    madvise(mapped, m->file_size, MADV_SEQUENTIAL);
#if defined(MADV_HUGEPAGE)
    madvise(mapped, m->file_size, MADV_HUGEPAGE);
#endif

    ByteCursor cur = { .base = m->mmap_base, .size = m->file_size, .pos = 0 };

    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!cursor_read_u32(&cur, &magic) || magic != GGUF_MAGIC) {
        fprintf(stderr, "[ERROR] Invalid GGUF magic bytes (0x%08X).\n", magic);
        goto fail;
    }
    if (!cursor_read_u32(&cur, &version) || (version != 2 && version != 3)) {
        fprintf(stderr, "[ERROR] Unsupported GGUF version: %u.\n", version);
        goto fail;
    }
    if (!cursor_read_u64(&cur, &n_tensors) || !cursor_read_u64(&cur, &n_kv)) {
        fprintf(stderr, "[ERROR] Corrupted GGUF header counts.\n");
        goto fail;
    }

    /* 1. Parse Metadata Key-Value Pairs */
    for (uint64_t i = 0; i < n_kv; ++i) {
        if (!parse_lfm2_metadata_kv(&cur, &m->cfg)) {
            fprintf(stderr, "[ERROR] Failed parsing metadata KV at index %llu.\n",
                    (unsigned long long)i);
            goto fail;
        }
    }

    /* 2. Parse Tensor Info Directory */
    m->n_tensors = n_tensors;
    m->tensors = (GGUFTensor *)calloc((size_t)n_tensors, sizeof(GGUFTensor));
    if (!m->tensors) goto fail;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensor *t = &m->tensors[i];
        if (!cursor_read_gguf_str(&cur, t->name, sizeof(t->name), NULL)) goto fail;
        if (!cursor_read_u32(&cur, &t->n_dims) || t->n_dims > LFM2_MAX_DIMS) goto fail;

        uint64_t total_elems = 1;
        for (uint32_t d = 0; d < LFM2_MAX_DIMS; ++d) {
            if (d < t->n_dims) {
                if (!cursor_read_u64(&cur, &t->ne[d])) goto fail;
                total_elems *= t->ne[d];
            } else {
                t->ne[d] = 1;
            }
        }
        if (!cursor_read_u32(&cur, &t->type) || !cursor_read_u64(&cur, &t->rel_offset)) {
            goto fail;
        }

        ggml_type_traits_t traits;
        if (get_ggml_type_traits(t->type, &traits) && traits.block_elements > 0) {
            t->bytes = (size_t)((total_elems / traits.block_elements) * traits.block_bytes);
        }
    }

    /* 3. Compute Aligned Tensor Data Base & Bind Zero-Copy Pointers */
    size_t align = m->cfg.alignment ? m->cfg.alignment : GGUF_DEFAULT_ALIGNMENT;
    size_t aligned_data_offset = (cur.pos + align - 1) & ~(align - 1);
    if (aligned_data_offset > m->file_size) {
        fprintf(stderr, "[ERROR] Tensor data offset exceeds file size.\n");
        goto fail;
    }
    m->tensor_data_base = m->mmap_base + aligned_data_offset;

    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensor *t = &m->tensors[i];
        size_t abs_off = aligned_data_offset + (size_t)t->rel_offset;
        if (abs_off + t->bytes > m->file_size) {
            fprintf(stderr, "[ERROR] Tensor %s exceeds EOF.\n", t->name);
            goto fail;
        }
        t->data = m->mmap_base + abs_off;
    }

    /* Sequential page touch warmup with dynamic page size (supporting 4 KB & Android 15+ 16 KB pages) */
    volatile uint64_t warmup_sink = 0;
    const uint8_t *warmup_ptr = m->mmap_base;
    long ps = sysconf(_SC_PAGESIZE);
    size_t page_step = (ps > 0) ? (size_t)ps : 4096;
    for (size_t off = 0; off < m->file_size; off += page_step) {
        warmup_sink += warmup_ptr[off];
    }
    (void)warmup_sink;

    if (!bind_lfm2_tensors(m)) goto fail;
    return true;

fail:
    if (m->tensors) free(m->tensors);
    if (m->mmap_base && m->mmap_base != MAP_FAILED) {
        munmap((void *)m->mmap_base, m->file_size);
    }
    if (m->fd >= 0) close(m->fd);
    m->fd = -1;
    m->mmap_base = NULL;
    m->tensors = NULL;
    return false;
}

bool lfm2_load_gguf(const char *filepath, LFM2Model *m) {
    if (!filepath) return false;
    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 24) {
        perror("fstat");
        close(fd);
        return false;
    }

    if (!lfm2_load_gguf_from_fd(fd, (size_t)st.st_size, m)) {
        return false;
    }
    return true;
}

void lfm2_free_model(LFM2Model *m) {
    if (m->tensors) free(m->tensors);
    if (m->mmap_base && m->mmap_base != MAP_FAILED) {
        munmap((void *)m->mmap_base, m->file_size);
    }
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
}

/* ============================================================================
 * 8. CLI Inspector & Pointer Verifier
 * ============================================================================ */
#ifndef LFM2_LOADER_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-lfm2-model.gguf>\n", argv[0]);
        return EXIT_FAILURE;
    }

    LFM2Model model;
    if (!lfm2_load_gguf(argv[1], &model)) {
        fprintf(stderr, "Failed to load GGUF model: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    printf("=== Liquid AI LFM2 / LFM2.5 / LFM3 GGUF Engine Inspector ===\n");
    printf("Architecture      : %s\n", model.cfg.arch);
    printf("File Size (mmap)  : %.2f MB\n", (double)model.file_size / (1024.0 * 1024.0));
    printf("Layers            : %u\n", model.cfg.n_layers);
    printf("Embedding Dim     : %u\n", model.cfg.n_embd);
    printf("FFN Hidden Dim    : %u\n", model.cfg.n_ff);
    printf("Vocab Size        : %u\n", model.cfg.n_vocab);
    printf("Context Length    : %u\n", model.cfg.n_ctx);
    printf("Query Heads       : %u (head_dim=%u)\n", model.cfg.n_head_q, model.cfg.head_dim);
    printf("ShortConv L-Cache : %u (ring buffer steps = %u)\n",
           model.cfg.shortconv_l_cache, model.cfg.shortconv_l_cache - 1);
    printf("RMSNorm Epsilon   : %g\n", model.cfg.rms_norm_eps);
    printf("RoPE Freq Base    : %.1f (SWA=%.1f, rope_dim=%u)\n",
           model.cfg.rope_freq_base, model.cfg.rope_freq_base_swa, model.cfg.rope_dim);
    if (model.cfg.sliding_window > 0) {
        printf("Sliding Window    : %u\n", model.cfg.sliding_window);
    }
    printf("Output Weight Tied: %s\n\n",
           (model.output == model.token_embd) ? "YES (tied to token_embd.weight)" : "NO");

    uint32_t n_conv = 0, n_attn = 0, n_swa = 0;
    printf("--- Hybrid Layer Routing Table ---\n");
    for (uint32_t l = 0; l < model.cfg.n_layers; ++l) {
        const LFM2Layer *ly = &model.layers[l];
        if (ly->block_type == LFM2_BLOCK_SHORTCONV) {
            n_conv++;
            ggml_type_traits_t tr;
            get_ggml_type_traits(ly->shortconv.in_proj->type, &tr);
            printf("Layer %2u : [SHORTCONV] in_proj=[%llu, %llu] (%s) | conv=[%llu, %llu] | ptr=%p\n",
                   l,
                   (unsigned long long)ly->shortconv.in_proj->ne[0],
                   (unsigned long long)ly->shortconv.in_proj->ne[1],
                   tr.name,
                   (unsigned long long)ly->shortconv.conv->ne[0],
                   (unsigned long long)ly->shortconv.conv->ne[1],
                   ly->shortconv.in_proj->data);
        } else {
            if (ly->block_type == LFM2_BLOCK_SWA_ATTN) n_swa++;
            else n_attn++;
            ggml_type_traits_t tr;
            get_ggml_type_traits(ly->attn.wq->type, &tr);
            printf("Layer %2u : [%s] kv_heads=%u | wq=[%llu, %llu] (%s) | q_norm=[%llu] | gate=%s | ptr=%p\n",
                   l,
                   (ly->block_type == LFM2_BLOCK_SWA_ATTN) ? "GQA_SWA  " : "GQA_FULL ",
                   model.cfg.n_head_kv[l],
                   (unsigned long long)ly->attn.wq->ne[0],
                   (unsigned long long)ly->attn.wq->ne[1],
                   tr.name,
                   (unsigned long long)(ly->attn.q_norm ? ly->attn.q_norm->ne[0] : 0),
                   ly->attn.gate ? "YES" : "NO",
                   ly->attn.wq->data);
        }
    }

    printf("\nSummary: %u ShortConv + %u SWA + %u Full Attention = %u total layers.\n",
           n_conv, n_swa, n_attn, model.cfg.n_layers);

    lfm2_free_model(&model);
    return EXIT_SUCCESS;
}
#endif

#endif /* LFM2_LOADER_INCLUDED */

