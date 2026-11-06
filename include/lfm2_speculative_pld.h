#ifndef LFM2_SPECULATIVE_PLD_H
#define LFM2_SPECULATIVE_PLD_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM2_PLD_MAX_DRAFT   7U                   /* Max K draft tokens -> Max Batch B = 8 */
#define LFM2_NGRAM_HT_SIZE   16384U               /* Power-of-2 L1-resident hash table     */
#define LFM2_NGRAM_HT_MASK   (LFM2_NGRAM_HT_SIZE - 1U)

/* ============================================================================
 * 1. O(1) Hierarchical 3-Gram / 2-Gram Prompt-Lookup Draft Engine
 * ============================================================================ */
typedef struct {
    uint32_t *history;                        /* [max_seq_len] full context token IDs */
    uint32_t  history_len;                    /* Total tokens currently in context    */
    uint32_t  max_seq_len;
    uint32_t  eos_token_id;                   /* Stop drafting at <|im_end|> (7)      */

    /* Maps hash(3-gram) or hash(2-gram) -> index of the NEXT token in `history[]` */
    int32_t   ht_3gram[LFM2_NGRAM_HT_SIZE];
    int32_t   ht_2gram[LFM2_NGRAM_HT_SIZE];

    /* Adaptive draft-length controller (shrinks K if miss streak occurs) */
    uint32_t  max_draft_k;                    /* Configured max K (1..7, default 6)   */
    uint32_t  cur_draft_k;                    /* Current adaptive K                   */
    uint32_t  consecutive_zero_accepts;

    /* Telemetry counters */
    uint64_t  total_steps;
    uint64_t  draft_batches_attempted;
    uint64_t  total_draft_tokens_proposed;
    uint64_t  total_draft_tokens_accepted;
} LFM2NgramIndex;

bool     lfm2_ngram_index_init(LFM2NgramIndex *idx, uint32_t max_seq_len,
                               uint32_t max_draft_k, uint32_t eos_token_id);
void     lfm2_ngram_index_reset(LFM2NgramIndex *idx);
void     lfm2_ngram_index_free(LFM2NgramIndex *idx);
void     lfm2_ngram_index_ingest(LFM2NgramIndex *idx, uint32_t token_id);
uint32_t lfm2_ngram_draft(const LFM2NgramIndex *idx,
                          uint32_t pending_token,
                          uint32_t draft_out[LFM2_PLD_MAX_DRAFT]);

#ifdef __cplusplus
}
#endif

#endif /* LFM2_SPECULATIVE_PLD_H */

