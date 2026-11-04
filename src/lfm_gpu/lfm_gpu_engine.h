#ifndef LFM_GPU_ENGINE_H
#define LFM_GPU_ENGINE_H

#include "lfm_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    LFMVulkanDevice      vk;
    VkDescriptorSetLayout desc_layout;
    VkDescriptorPool     desc_pool;
    VkDescriptorSet      desc_set;

    VkPipelineLayout     pipe_layout;
    VkPipeline           pipe_gemv_f32;
    VkPipeline           pipe_gemv_q4_k;

    /* Shared host-coherent buffers for zero-copy UMA GPU access */
    LFMVulkanBuffer      buf_w;
    LFMVulkanBuffer      buf_x;
    LFMVulkanBuffer      buf_out;

    size_t               max_w_size;
    size_t               max_x_size;
    size_t               max_out_size;
} LFMGPUEngine;

/* Initializes GPU engine.
 * target_device_index:
 *   -1: Auto-select Intel integrated GPU (0x8086)
 *    0: Select first GPU (e.g. NVIDIA MX250 if present)
 *    1: Select second GPU (e.g. Intel UHD Graphics) */
bool lfm_gpu_engine_init(LFMGPUEngine *gpu, int target_device_index, size_t max_matrix_bytes);
void lfm_gpu_engine_free(LFMGPUEngine *gpu);

/* Runs GPU GEMV for FP32 matrix: out[r] = dot(W_row[r], x) */
bool lfm_gpu_gemv_f32(LFMGPUEngine *gpu, float *out, const float *W, const float *x,
                      uint32_t n_cols, uint32_t n_rows);

/* Runs GPU GEMV for Q4_K quantized matrix directly from GGUF mmap */
bool lfm_gpu_gemv_q4_k(LFMGPUEngine *gpu, float *out, const void *w_q4_data, const float *x,
                       uint32_t n_cols, uint32_t n_rows);

#ifdef __cplusplus
}
#endif

#endif /* LFM_GPU_ENGINE_H */
