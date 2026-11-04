#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Include the GGUF loader from Step 1 (excluding its demo main) if not already included */
#ifndef LFM2_LOADER_INCLUDED
#ifndef LFM2_LOADER_NO_MAIN
#define LFM2_LOADER_NO_MAIN 1
#endif
#define main lfm2_loader_demo_main
#if defined(__has_include)
#if __has_include("lfm2_loader.c")
#include "lfm2_loader.c"
#elif __has_include("tools/lfm2_loader.c")
#include "tools/lfm2_loader.c"
#endif
#else
#include "lfm2_loader.c"
#endif
#undef main
#endif

#ifndef LFM2_BPE_INCLUDED
#define LFM2_BPE_INCLUDED 1

#define LFM2_BPE_MAX_WORD_BYTES 512
#define LFM2_BPE_MAX_HEAP_SIZE  1024

/* ============================================================================
 * 1. Hash Table Structures for O(1) String -> ID and Pair -> (Rank, MergedID)
 * ============================================================================ */
typedef struct {
    uint64_t hash;      /* 64-bit FNV-1a hash of token string          */
    uint32_t token_id;  /* Index in v->tokens (UINT32_MAX = empty)     */
    uint32_t str_len;   /* Byte length of v->tokens[token_id]          */
} TokenStrSlot;

typedef struct {
    uint64_t pair_key;  /* ((uint64_t)left_id << 32) | right_id        */
    uint32_t rank;      /* Index in tokenizer.ggml.merges (0 = highest)*/
    uint32_t merged_id; /* Token ID of concatenated "leftright" token  */
} BPEMergeSlot;

typedef struct {
    char    *str;
    uint32_t len;
    uint32_t id;
} SpecialToken;

typedef struct {
    uint32_t      n_vocab;
    uint32_t      n_merges;
    char        **tokens;          /* [n_vocab] null-terminated strings      */
    uint32_t     *token_lens;      /* [n_vocab] byte lengths                 */

    /* Open-addressing hash table: string slice -> token_id */
    uint32_t      str_ht_cap;      /* Power of 2 >= 2 * n_vocab              */
    TokenStrSlot *str_ht;

    /* Open-addressing hash table: (left_id, right_id) -> (rank, merged_id) */
    uint32_t      merge_ht_cap;    /* Power of 2 >= 2 * n_merges             */
    BPEMergeSlot *merge_ht;

    /* Fast O(1) raw byte (0..255) -> initial BPE symbol token_id */
    uint32_t      byte_to_id[256];

    /* Special control tokens (<|startoftext|>, <|im_start|>, <|im_end|>, etc.) */
    SpecialToken  specials[64];
    uint32_t      n_specials;

    uint32_t      id_pad;
    uint32_t      id_bos;
    uint32_t      id_im_start;
    uint32_t      id_im_end;
    uint32_t      id_tool_start;
    uint32_t      id_tool_end;
} LFM2Vocab;

/* ============================================================================
 * 2. Hash Functions & Lookup Primitives
 * ============================================================================ */
static inline uint64_t fnv1a_hash_bytes(const char *s, size_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static inline uint64_t hash_u64_pair(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

static uint32_t next_pow2_u32(uint32_t x) {
    uint32_t p = 1024;
    while (p < x) p <<= 1;
    return p;
}

static void vocab_str_ht_insert(LFM2Vocab *v, const char *s, uint32_t len, uint32_t id) {
    if (v->str_ht_cap == 0) return;
    const uint64_t h    = fnv1a_hash_bytes(s, len);
    const uint32_t mask = v->str_ht_cap - 1;
    uint32_t slot       = (uint32_t)h & mask;

    while (v->str_ht[slot].token_id != UINT32_MAX) {
        slot = (slot + 1) & mask;
    }
    v->str_ht[slot].hash     = h;
    v->str_ht[slot].token_id = id;
    v->str_ht[slot].str_len  = len;
}

static uint32_t vocab_lookup_str(const LFM2Vocab *v, const char *s, size_t len) {
    if (v->str_ht_cap == 0) return UINT32_MAX;
    const uint64_t h    = fnv1a_hash_bytes(s, len);
    const uint32_t mask = v->str_ht_cap - 1;
    uint32_t slot       = (uint32_t)h & mask;

    while (v->str_ht[slot].token_id != UINT32_MAX) {
        if (v->str_ht[slot].hash == h && v->str_ht[slot].str_len == len) {
            const uint32_t tid = v->str_ht[slot].token_id;
            if (memcmp(v->tokens[tid], s, len) == 0) {
                return tid;
            }
        }
        slot = (slot + 1) & mask;
    }
    return UINT32_MAX;
}

static void vocab_merge_ht_insert(LFM2Vocab *v, uint32_t left_id, uint32_t right_id,
                                  uint32_t rank, uint32_t merged_id) {
    if (v->merge_ht_cap == 0) return;
    const uint64_t key  = ((uint64_t)left_id << 32) | (uint64_t)right_id;
    const uint64_t h    = hash_u64_pair(key);
    const uint32_t mask = v->merge_ht_cap - 1;
    uint32_t slot       = (uint32_t)h & mask;

    while (v->merge_ht[slot].pair_key != UINT64_MAX) {
        if (v->merge_ht[slot].pair_key == key) return; /* Keep earliest rank */
        slot = (slot + 1) & mask;
    }
    v->merge_ht[slot].pair_key  = key;
    v->merge_ht[slot].rank      = rank;
    v->merge_ht[slot].merged_id = merged_id;
}

static inline bool vocab_lookup_merge(const LFM2Vocab *v, uint32_t left_id, uint32_t right_id,
                                      uint32_t *rank_out, uint32_t *merged_id_out) {
    if (v->merge_ht_cap == 0) return false;
    const uint64_t key  = ((uint64_t)left_id << 32) | (uint64_t)right_id;
    const uint64_t h    = hash_u64_pair(key);
    const uint32_t mask = v->merge_ht_cap - 1;
    uint32_t slot       = (uint32_t)h & mask;

    while (v->merge_ht[slot].pair_key != UINT64_MAX) {
        if (v->merge_ht[slot].pair_key == key) {
            *rank_out      = v->merge_ht[slot].rank;
            *merged_id_out = v->merge_ht[slot].merged_id;
            return true;
        }
        slot = (slot + 1) & mask;
    }
    return false;
}

/* ============================================================================
 * 3. GPT-2 Byte-to-Unicode Mapping for Precomputing byte_to_id[256]
 * ============================================================================ */
static size_t gpt2_single_byte_to_utf8(uint8_t b, char out[4]) {
    bool is_direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174);
    uint16_t cp = is_direct ? b : (uint16_t)(256 + (
        (b < 33) ? b :
        (b >= 127 && b <= 160) ? (b - 127 + 33) : 67
    ));
    if (cp < 0x80) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    }
}

/* ============================================================================
 * 4. GGUF Vocabulary & BPE Merges Loader (Order-Independent Two-Phase Parse)
 * ============================================================================ */
bool lfm2_vocab_load(LFM2Vocab *v, const LFM2Model *m) {
    memset(v, 0, sizeof(*v));
    v->id_pad      = 0;
    v->id_bos      = 1;
    v->id_im_start = 6;
    v->id_im_end   = 7;

    for (int b = 0; b < 256; ++b) {
        v->byte_to_id[b] = UINT32_MAX;
    }

    ByteCursor cur = { .base = m->mmap_base, .size = m->file_size, .pos = 0 };
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!cursor_read_u32(&cur, &magic) || !cursor_read_u32(&cur, &version) ||
        !cursor_read_u64(&cur, &n_tensors) || !cursor_read_u64(&cur, &n_kv)) {
        return false;
    }

    size_t tokens_offset = 0;
    size_t merges_offset = 0;

    /* Pass 1: Locate offsets of tokenizer.ggml.tokens and tokenizer.ggml.merges */
    for (uint64_t i = 0; i < n_kv; ++i) {
        char key[128];
        uint32_t val_type = 0;
        if (!cursor_read_gguf_str(&cur, key, sizeof(key), NULL)) return false;
        if (!cursor_read_u32(&cur, &val_type)) return false;

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && val_type == GGUF_TYPE_ARRAY) {
            tokens_offset = cur.pos;
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && val_type == GGUF_TYPE_ARRAY) {
            merges_offset = cur.pos;
        }
        if (!cursor_skip_gguf_value(&cur, val_type)) return false;
    }

    if (tokens_offset == 0) {
        fprintf(stderr, "[ERROR] GGUF missing tokenizer.ggml.tokens array.\n");
        return false;
    }

    /* Pass 2: Load all token strings and populate TokenStrHash */
    cur.pos = tokens_offset;
    uint32_t tok_elem_type = 0;
    uint64_t tok_count = 0;
    if (!cursor_read_u32(&cur, &tok_elem_type) || !cursor_read_u64(&cur, &tok_count) ||
        tok_elem_type != GGUF_TYPE_STRING) {
        return false;
    }

    v->n_vocab    = (uint32_t)tok_count;
    v->tokens     = (char **)calloc(v->n_vocab, sizeof(char *));
    v->token_lens = (uint32_t *)calloc(v->n_vocab, sizeof(uint32_t));
    v->str_ht_cap = next_pow2_u32(v->n_vocab * 2);
    v->str_ht     = (TokenStrSlot *)malloc((size_t)v->str_ht_cap * sizeof(TokenStrSlot));
    if (!v->tokens || !v->token_lens || !v->str_ht) return false;

    for (uint32_t i = 0; i < v->str_ht_cap; ++i) {
        v->str_ht[i].token_id = UINT32_MAX;
    }

    for (uint32_t tid = 0; tid < v->n_vocab; ++tid) {
        uint64_t slen = 0;
        if (!cursor_read_u64(&cur, &slen) || cur.pos + slen > cur.size) return false;

        v->tokens[tid] = (char *)malloc((size_t)slen + 1);
        if (!v->tokens[tid]) return false;
        memcpy(v->tokens[tid], cur.base + cur.pos, (size_t)slen);
        v->tokens[tid][slen] = '\0';
        v->token_lens[tid]   = (uint32_t)slen;
        cur.pos += (size_t)slen;

        vocab_str_ht_insert(v, v->tokens[tid], (uint32_t)slen, tid);

        /* Register special tokens for exact pre-splitting */
        const char *tok = v->tokens[tid];
        bool is_special = false;
        if (slen >= 5 && tok[0] == '<' && tok[1] == '|' &&
            tok[slen - 2] == '|' && tok[slen - 1] == '>') {
            is_special = true;
        } else if (strcmp(tok, "<s>") == 0 || strcmp(tok, "</s>") == 0 || strcmp(tok, "<unk>") == 0) {
            is_special = true;
        }

        if (is_special && v->n_specials < 64) {
            v->specials[v->n_specials++] = (SpecialToken){
                .str = v->tokens[tid], .len = (uint32_t)slen, .id = tid
            };
        }

        if (strcmp(tok, "<|pad|>") == 0)                   v->id_pad        = tid;
        else if (strcmp(tok, "<|startoftext|>") == 0)      v->id_bos        = tid;
        else if (strcmp(tok, "<s>") == 0)                  v->id_bos        = tid;
        else if (strcmp(tok, "<|im_start|>") == 0)         v->id_im_start   = tid;
        else if (strcmp(tok, "<|im_end|>") == 0)           v->id_im_end     = tid;
        else if (strcmp(tok, "</s>") == 0)                 v->id_im_end     = tid;
        else if (strcmp(tok, "<|tool_call_start|>") == 0)  v->id_tool_start = tid;
        else if (strcmp(tok, "<|tool_call_end|>") == 0)    v->id_tool_end   = tid;
    }

    /* Populate O(1) raw byte -> initial token_id lookup table */
    for (int b = 0; b < 256; ++b) {
        char u8[4];
        size_t ulen = gpt2_single_byte_to_utf8((uint8_t)b, u8);
        v->byte_to_id[b] = vocab_lookup_str(v, u8, ulen);
    }

    /* Pass 3: Load tokenizer.ggml.merges and build BPEMergeSlot Hash Table */
    if (merges_offset != 0) {
        cur.pos = merges_offset;
        uint32_t m_elem_type = 0;
        uint64_t m_count = 0;
        if (!cursor_read_u32(&cur, &m_elem_type) || !cursor_read_u64(&cur, &m_count) ||
            m_elem_type != GGUF_TYPE_STRING) {
            return false;
        }

        v->n_merges     = (uint32_t)m_count;
        v->merge_ht_cap = next_pow2_u32(v->n_merges * 2);
        v->merge_ht     = (BPEMergeSlot *)malloc((size_t)v->merge_ht_cap * sizeof(BPEMergeSlot));
        if (!v->merge_ht) return false;

        for (uint32_t i = 0; i < v->merge_ht_cap; ++i) {
            v->merge_ht[i].pair_key = UINT64_MAX;
        }

        char merged_buf[512];
        for (uint32_t rank = 0; rank < v->n_merges; ++rank) {
            uint64_t mlen = 0;
            if (!cursor_read_u64(&cur, &mlen) || cur.pos + mlen > cur.size) return false;

            const char *mstr = (const char *)(cur.base + cur.pos);
            cur.pos += (size_t)mlen;

            /* Find the space separating "left_tok right_tok" */
            const char *space = (const char *)memchr(mstr, ' ', (size_t)mlen);
            if (!space) continue;

            const size_t left_len  = (size_t)(space - mstr);
            const size_t right_len = (size_t)mlen - left_len - 1;
            if (left_len + right_len >= sizeof(merged_buf)) continue;

            const uint32_t left_id  = vocab_lookup_str(v, mstr, left_len);
            const uint32_t right_id = vocab_lookup_str(v, space + 1, right_len);

            memcpy(merged_buf, mstr, left_len);
            memcpy(merged_buf + left_len, space + 1, right_len);
            const uint32_t merged_id = vocab_lookup_str(v, merged_buf, left_len + right_len);

            if (left_id != UINT32_MAX && right_id != UINT32_MAX && merged_id != UINT32_MAX) {
                vocab_merge_ht_insert(v, left_id, right_id, rank, merged_id);
            }
        }
    }

    return true;
}

void lfm2_vocab_free(LFM2Vocab *v) {
    if (v->tokens) {
        for (uint32_t i = 0; i < v->n_vocab; ++i) free(v->tokens[i]);
        free(v->tokens);
    }
    free(v->token_lens);
    free(v->str_ht);
    free(v->merge_ht);
    memset(v, 0, sizeof(*v));
}

/* ============================================================================
 * 5. Exact O(N log N) BPE Merge Engine (Doubly-Linked List + Binary Min-Heap)
 * ============================================================================ */
typedef struct {
    uint32_t token_id; /* Current symbol token ID (UINT32_MAX if merged away) */
    int32_t  prev;     /* Index of previous active symbol (-1 if head)        */
    int32_t  next;     /* Index of next active symbol (-1 if tail)            */
    uint32_t ver;      /* Incremented on each merge to invalidate stale heap  */
} BPESymbol;

typedef struct {
    uint32_t rank;      /* Priority from tokenizer.ggml.merges (lower = 1st)  */
    uint32_t merged_id; /* Resulting token ID if merged                       */
    int32_t  left;      /* Left symbol index in syms[]                        */
    int32_t  right;     /* Right symbol index in syms[]                       */
    uint32_t left_ver;  /* Expected syms[left].ver                            */
    uint32_t right_ver; /* Expected syms[right].ver                           */
} BPEHeapItem;

typedef struct {
    BPEHeapItem items[LFM2_BPE_MAX_HEAP_SIZE];
    uint32_t    size;
} BPEMinHeap;

static inline bool bpe_item_less(const BPEHeapItem *a, const BPEHeapItem *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->left < b->left; /* Tie-break leftmost pair first */
}

static void bpe_heap_push(BPEMinHeap *h, BPEHeapItem item) {
    if (h->size >= LFM2_BPE_MAX_HEAP_SIZE) return;
    uint32_t i = h->size++;
    h->items[i] = item;
    while (i > 0) {
        uint32_t parent = (i - 1) >> 1;
        if (!bpe_item_less(&h->items[i], &h->items[parent])) break;
        BPEHeapItem tmp  = h->items[parent];
        h->items[parent] = h->items[i];
        h->items[i]      = tmp;
        i = parent;
    }
}

static BPEHeapItem bpe_heap_pop(BPEMinHeap *h) {
    BPEHeapItem top = h->items[0];
    h->items[0] = h->items[--h->size];
    uint32_t i = 0;
    while (1) {
        uint32_t l = 2 * i + 1;
        uint32_t r = 2 * i + 2;
        uint32_t smallest = i;
        if (l < h->size && bpe_item_less(&h->items[l], &h->items[smallest])) smallest = l;
        if (r < h->size && bpe_item_less(&h->items[r], &h->items[smallest])) smallest = r;
        if (smallest == i) break;
        BPEHeapItem tmp    = h->items[i];
        h->items[i]        = h->items[smallest];
        h->items[smallest] = tmp;
        i = smallest;
    }
    return top;
}

static inline void bpe_try_enqueue_pair(const LFM2Vocab *v, BPEMinHeap *heap,
                                        const BPESymbol *syms, int32_t left, int32_t right) {
    if (left < 0 || right < 0) return;
    uint32_t rank = 0, merged_id = 0;
    if (vocab_lookup_merge(v, syms[left].token_id, syms[right].token_id, &rank, &merged_id)) {
        bpe_heap_push(heap, (BPEHeapItem){
            .rank      = rank,
            .merged_id = merged_id,
            .left      = left,
            .right     = right,
            .left_ver  = syms[left].ver,
            .right_ver = syms[right].ver
        });
    }
}

/* Runs exact BPE rank merging on a single pre-tokenized word [bytes, len] */
static uint32_t bpe_encode_word(const LFM2Vocab *v, const uint8_t *bytes, size_t len,
                                uint32_t *out_ids, uint32_t max_ids) {
    if (len == 0 || max_ids == 0) return 0;
    if (len > LFM2_BPE_MAX_WORD_BYTES) len = LFM2_BPE_MAX_WORD_BYTES;

    /* Check if all bytes in this word have valid initial token mappings */
    bool has_all_bytes = true;
    for (size_t i = 0; i < len; ++i) {
        if (v->byte_to_id[bytes[i]] == UINT32_MAX) {
            has_all_bytes = false;
            break;
        }
    }

    /* Fallback to greedy string lookup if no BPE merges loaded or missing byte tokens */
    if (!has_all_bytes || v->n_merges == 0) {
        size_t off = 0;
        uint32_t n_out = 0;
        while (off < len && n_out < max_ids) {
            uint32_t best_id = UINT32_MAX;
            size_t best_len = 0;
            size_t max_match = (len - off < 64) ? (len - off) : 64;
            for (size_t l = max_match; l >= 1; --l) {
                uint32_t tid = vocab_lookup_str(v, (const char *)(bytes + off), l);
                if (tid != UINT32_MAX) {
                    best_id = tid;
                    best_len = l;
                    break;
                }
            }
            if (best_id != UINT32_MAX) {
                out_ids[n_out++] = best_id;
                off += best_len;
            } else {
                off++;
            }
        }
        return n_out;
    }

    BPESymbol syms[LFM2_BPE_MAX_WORD_BYTES];
    for (size_t i = 0; i < len; ++i) {
        syms[i].token_id = v->byte_to_id[bytes[i]];
        syms[i].prev     = (int32_t)i - 1;
        syms[i].next     = (i + 1 < len) ? (int32_t)(i + 1) : -1;
        syms[i].ver      = 0;
    }

    BPEMinHeap heap = { .size = 0 };
    for (size_t i = 0; i + 1 < len; ++i) {
        bpe_try_enqueue_pair(v, &heap, syms, (int32_t)i, (int32_t)(i + 1));
    }

    while (heap.size > 0) {
        BPEHeapItem top = bpe_heap_pop(&heap);
        const int32_t L = top.left;
        const int32_t R = top.right;

        /* Skip if either node was already merged or modified */
        if (syms[L].token_id == UINT32_MAX || syms[R].token_id == UINT32_MAX) continue;
        if (syms[L].ver != top.left_ver || syms[R].ver != top.right_ver)       continue;
        if (syms[L].next != R)                                                 continue;

        /* Merge R into L */
        syms[L].token_id = top.merged_id;
        syms[L].ver++;
        syms[R].token_id = UINT32_MAX;

        const int32_t next_idx = syms[R].next;
        syms[L].next = next_idx;
        if (next_idx >= 0) {
            syms[next_idx].prev = L;
        }

        /* Enqueue newly formed neighbor bigrams */
        bpe_try_enqueue_pair(v, &heap, syms, syms[L].prev, L);
        bpe_try_enqueue_pair(v, &heap, syms, L, syms[L].next);
    }

    uint32_t n_out = 0;
    for (int32_t idx = 0; idx >= 0 && n_out < max_ids; idx = syms[idx].next) {
        if (syms[idx].token_id != UINT32_MAX) {
            out_ids[n_out++] = syms[idx].token_id;
        }
    }
    return n_out;
}

/* ============================================================================
 * 6. GPT-2 / Llama Pre-Tokenizer State Machine + Special-Tag Splitter
 * ============================================================================ */
static inline bool is_ascii_letter(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0x80);
}
static inline bool is_ascii_digit(uint8_t c) {
    return (c >= '0' && c <= '9');
}
static inline bool is_whitespace(uint8_t c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f');
}

/* Computes the byte length of the next GPT-2 pre-token starting at `s[0..len-1]` */
static size_t next_gpt2_pretoken_len(const uint8_t *s, size_t len) {
    if (len == 0) return 0;
    size_t i = 0;

    /* 1. English contractions: 's, 't, 're, 've, 'm, 'll, 'd */
    if (s[0] == '\'' && len >= 2) {
        uint8_t c1 = s[1] | 0x20;
        if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
        if (len >= 3) {
            uint8_t c2 = s[2] | 0x20;
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                return 3;
            }
        }
    }

    /* 2. Optional single leading space before a word, number, or punctuation */
    if (s[0] == ' ' && len > 1 && !is_whitespace(s[1])) {
        i = 1;
    }

    if (is_ascii_letter(s[i])) {
        while (i < len && is_ascii_letter(s[i])) i++;
        return i;
    }
    if (is_ascii_digit(s[i])) {
        /* Group up to 3 digits per pre-token (Llama-3 / LFM2 number splitting) */
        size_t digits = 0;
        while (i < len && is_ascii_digit(s[i]) && digits < 3) {
            i++;
            digits++;
        }
        return i;
    }
    if (!is_whitespace(s[i])) {
        /* Contiguous non-whitespace, non-alphanumeric punctuation */
        while (i < len && !is_whitespace(s[i]) && !is_ascii_letter(s[i]) &&
               !is_ascii_digit(s[i]) && s[i] != '\'') {
            i++;
        }
        return (i > 0) ? i : 1;
    }

    /* 3. Whitespace runs (keep newline sequences or trailing spaces grouped) */
    if (s[0] == '\n' || s[0] == '\r') {
        while (i < len && (s[i] == '\n' || s[i] == '\r')) i++;
        return i;
    }
    while (i < len && is_whitespace(s[i]) && s[i] != '\n' && s[i] != '\r') {
        /* Leave the last space attached to the following word if one exists */
        if (i + 1 < len && s[i] == ' ' && !is_whitespace(s[i + 1]) && i > 0) {
            break;
        }
        i++;
    }
    return (i > 0) ? i : 1;
}

uint32_t lfm2_tokenize(const LFM2Vocab *v, const char *text,
                       uint32_t *out_ids, uint32_t max_ids) {
    uint32_t n_out = 0;
    const char *p = text;

    while (*p && n_out < max_ids) {
        /* 1. Check if cursor begins with any registered special token */
        int64_t matched_id = -1;
        uint32_t matched_len = 0;
        for (uint32_t sp = 0; sp < v->n_specials; ++sp) {
            const uint32_t slen = v->specials[sp].len;
            if (slen > matched_len && strncmp(p, v->specials[sp].str, slen) == 0) {
                matched_id  = (int64_t)v->specials[sp].id;
                matched_len = slen;
            }
        }
        if (matched_id >= 0) {
            out_ids[n_out++] = (uint32_t)matched_id;
            p += matched_len;
            continue;
        }

        /* 2. Find the end of the current plain-text span (up to next special token) */
        size_t span_len = strlen(p);
        for (uint32_t sp = 0; sp < v->n_specials; ++sp) {
            const char *pos = strstr(p, v->specials[sp].str);
            if (pos && (size_t)(pos - p) < span_len) {
                span_len = (size_t)(pos - p);
            }
        }
        if (span_len == 0) {
            n_out += bpe_encode_word(v, (const uint8_t *)p, 1, out_ids + n_out, max_ids - n_out);
            p += 1;
            continue;
        }

        /* 3. Split plain-text span into GPT-2 pre-tokens and run BPE merge on each */
        size_t offset = 0;
        while (offset < span_len && n_out < max_ids) {
            size_t piece_len = next_gpt2_pretoken_len((const uint8_t *)(p + offset),
                                                      span_len - offset);
            n_out += bpe_encode_word(v, (const uint8_t *)(p + offset), piece_len,
                                     out_ids + n_out, max_ids - n_out);
            offset += piece_len;
        }
        p += span_len;
    }

    return n_out;
}

/* Decodes a single token ID back into raw UTF-8 bytes */
size_t lfm2_detokenize_token(const LFM2Vocab *v, uint32_t tid, char *out_buf, size_t out_cap) {
    if (tid >= v->n_vocab || out_cap == 0) return 0;
    const uint8_t *s = (const uint8_t *)v->tokens[tid];
    size_t pos = 0;

    while (*s && pos + 1 < out_cap) {
        if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
            uint16_t cp = ((uint16_t)(s[0] & 0x1F) << 6) | (uint16_t)(s[1] & 0x3F);
            if (cp >= 256 && cp <= 323) {
                uint16_t off = cp - 256;
                uint8_t raw = (off < 33) ? (uint8_t)off :
                              (off < 67) ? (uint8_t)(off - 33 + 127) : 173;
                out_buf[pos++] = (char)raw;
                s += 2;
                continue;
            }
        }
        out_buf[pos++] = (char)(*s++);
    }
    out_buf[pos] = '\0';
    return pos;
}

/* Decodes a token and writes to stdout */
void lfm2_print_token(const LFM2Vocab *v, uint32_t tid) {
    if (tid >= v->n_vocab || tid == v->id_bos || tid == v->id_im_end || tid == v->id_pad) {
        return;
    }
    char buf[512];
    size_t len = lfm2_detokenize_token(v, tid, buf, sizeof(buf));
    if (len > 0) {
        fwrite(buf, 1, len, stdout);
        fflush(stdout);
    }
}

#endif /* LFM2_BPE_INCLUDED */

/* ============================================================================
 * 7. Standalone CLI Verification & Round-Trip Test
 * ============================================================================ */
#ifdef LFM2_BPE_STANDALONE_TEST
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path-to-lfm2.gguf> [prompt]\n", argv[0]);
        return EXIT_FAILURE;
    }

    LFM2Model model;
    if (!lfm2_load_gguf(argv[1], &model)) return EXIT_FAILURE;

    LFM2Vocab vocab;
    if (!lfm2_vocab_load(&vocab, &model)) {
        fprintf(stderr, "Failed to load GGUF vocabulary & merges.\n");
        lfm2_free_model(&model);
        return EXIT_FAILURE;
    }

    const char *test_prompt = (argc >= 3) ? argv[2] :
        "<|startoftext|><|im_start|>system\nYou are a helpful assistant trained by Liquid AI.<|im_end|>\n"
        "<|im_start|>user\nWrite a fast C11 function `vec_dot_q4_K` that doesn't allocate heap memory!<|im_end|>\n"
        "<|im_start|>assistant\n";

    uint32_t ids[512];
    uint32_t n_ids = lfm2_tokenize(&vocab, test_prompt, ids, 512);

    printf("=== LFM2 Exact GGUF BPE Tokenizer & Merge Inspector ===\n");
    printf("Vocab Size     : %u tokens (hash table cap = %u)\n", vocab.n_vocab, vocab.str_ht_cap);
    printf("BPE Merges     : %u rules  (hash table cap = %u)\n", vocab.n_merges, vocab.merge_ht_cap);
    printf("Special Tokens : %u registered (BOS=%u, IM_START=%u, IM_END=%u)\n\n",
           vocab.n_specials, vocab.id_bos, vocab.id_im_start, vocab.id_im_end);

    printf("Encoded %u tokens:\n", n_ids);
    for (uint32_t i = 0; i < n_ids; ++i) {
        char decoded[128];
        lfm2_detokenize_token(&vocab, ids[i], decoded, sizeof(decoded));
        printf("  [%3u] id=%6u | raw=\"%s\" | decoded=\"%s\"\n",
               i, ids[i], vocab.tokens[ids[i]], decoded);
    }

    lfm2_vocab_free(&vocab);
    lfm2_free_model(&model);
    return EXIT_SUCCESS;
}
#endif
