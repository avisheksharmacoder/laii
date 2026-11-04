#ifndef LFM_NV_ENGINE_H
#define LFM_NV_ENGINE_H

#include "lfm_nv_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    LFMNVDevice           vk;
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout      pipe_layout;
    VkDescriptorPool      desc_pool;
    VkDescriptorSet       desc_set;
    VkPipeline            pipe_q4_k;

    LFMVulkanBuffer_NV    buf_w_shared; /* Default VRAM scratch buffer for weights */
    LFMVulkanBuffer_NV    buf_x;        /* Activation vector (Host-visible) */
    LFMVulkanBuffer_NV    buf_out;      /* Output vector (Host-visible) */

    size_t                max_w_size;
    size_t                max_x_size;
    size_t                max_out_size;
} LFMNVEngine;

/* Initialize NVIDIA Engine with dedicated GDDR5 VRAM */
bool lfm_nv_engine_init(LFMNVEngine *gpu, size_t max_matrix_bytes);
void lfm_nv_engine_free(LFMNVEngine *gpu);

/* Uploads a weight tensor into a persistent GDDR5 VRAM buffer */
bool lfm_nv_engine_alloc_and_upload_vram(LFMNVEngine *gpu, LFMVulkanBuffer_NV *vram_buf,
                                        const void *w_host, size_t w_bytes);

/* Executes Q4_K GEMV where weights are ALREADY RESIDENT in GDDR5 VRAM (Zero-copy GDDR5 read at full bandwidth) */
bool lfm_nv_engine_gemv_q4_k_vram(LFMNVEngine *gpu, float *out, const LFMVulkanBuffer_NV *vram_buf,
                                  const float *x, uint32_t n_cols, uint32_t n_rows);

/* Fallback: uploads on-the-fly to scratch VRAM and executes */
bool lfm_nv_engine_gemv_q4_k(LFMNVEngine *gpu, float *out, const void *w_data,
                             const float *x, uint32_t n_cols, uint32_t n_rows);

#ifdef __cplusplus
}
#endif

#endif /* LFM_NV_ENGINE_H */
