#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "lfm_nv_engine.h"
#include "gemv_q4_k_nv_spv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_cols;
    uint32_t n_rows;
} PushConstantsNV;

static VkPipeline create_compute_pipeline(LFMNVDevice *dev, VkPipelineLayout layout,
                                          const unsigned char *spv_code, size_t spv_size) {
    VkShaderModuleCreateInfo sInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_size,
        .pCode = (const uint32_t *)spv_code
    };

    VkShaderModule module = 0;
    if (dev->fn.vkCreateShaderModule(dev->device, &sInfo, NULL, &module) != VK_SUCCESS) {
        return 0;
    }

    VkComputePipelineCreateInfo pInfo = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main"
        },
        .layout = layout
    };

    VkPipeline pipeline = 0;
    VkResult res = dev->fn.vkCreateComputePipelines(dev->device, NULL, 1, &pInfo, NULL, &pipeline);
    dev->fn.vkDestroyShaderModule(dev->device, module, NULL);

    return (res == VK_SUCCESS) ? pipeline : 0;
}

bool lfm_nv_engine_init(LFMNVEngine *gpu, size_t max_matrix_bytes) {
    if (!gpu) return false;
    memset(gpu, 0, sizeof(*gpu));

    if (!lfm_nv_init(&gpu->vk)) {
        return false;
    }

    LFMNVDevice *dev = &gpu->vk;

    /* 1. Create Descriptor Set Layout (Bindings 0: W in VRAM, 1: X host, 2: Out host) */
    VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };

    VkDescriptorSetLayoutCreateInfo dslInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings
    };

    if (dev->fn.vkCreateDescriptorSetLayout(dev->device, &dslInfo, NULL, &gpu->desc_layout) != VK_SUCCESS) {
        lfm_nv_engine_free(gpu);
        return false;
    }

    /* 2. Create Pipeline Layout with Push Constants */
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstantsNV)
    };

    VkPipelineLayoutCreateInfo plInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &gpu->desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    if (dev->fn.vkCreatePipelineLayout(dev->device, &plInfo, NULL, &gpu->pipe_layout) != VK_SUCCESS) {
        lfm_nv_engine_free(gpu);
        return false;
    }

    /* 3. Create NVIDIA Q4_K Compute Pipeline */
    gpu->pipe_q4_k = create_compute_pipeline(dev, gpu->pipe_layout,
                                             src_lfm_nv_shaders_gemv_q4_k_nv_spv,
                                             src_lfm_nv_shaders_gemv_q4_k_nv_spv_len);
    if (!gpu->pipe_q4_k) {
        fprintf(stderr, "[LFM-NV ERROR] Failed to create Q4_K compute pipeline.\n");
        lfm_nv_engine_free(gpu);
        return false;
    }

    /* 4. Allocate Descriptor Pool */
    VkDescriptorPoolSize poolSizes[1] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 32 }
    };

    VkDescriptorPoolCreateInfo dpInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 16,
        .poolSizeCount = 1,
        .pPoolSizes = poolSizes
    };

    if (dev->fn.vkCreateDescriptorPool(dev->device, &dpInfo, NULL, &gpu->desc_pool) != VK_SUCCESS) {
        lfm_nv_engine_free(gpu);
        return false;
    }

    VkDescriptorSetAllocateInfo dsAlloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gpu->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gpu->desc_layout
    };

    if (dev->fn.vkAllocateDescriptorSets(dev->device, &dsAlloc, &gpu->desc_set) != VK_SUCCESS) {
        lfm_nv_engine_free(gpu);
        return false;
    }

    /* 5. Allocate Buffers: W in 2GB GDDR5 VRAM, X/Out in host-visible memory */
    size_t w_bytes = (max_matrix_bytes > 0) ? max_matrix_bytes : (64 * 1024 * 1024);
    size_t x_bytes = 16 * 1024 * sizeof(float);
    size_t out_bytes = 65536 * sizeof(float);

    gpu->max_w_size   = w_bytes;
    gpu->max_x_size   = x_bytes;
    gpu->max_out_size = out_bytes;

    if (!lfm_nv_buffer_alloc_vram(dev, &gpu->buf_w_shared, w_bytes)) {
        fprintf(stderr, "[LFM-NV ERROR] Failed to allocate scratch GDDR5 VRAM buffer.\n");
        lfm_nv_engine_free(gpu);
        return false;
    }

    if (!lfm_nv_buffer_alloc_host(dev, &gpu->buf_x, x_bytes) ||
        !lfm_nv_buffer_alloc_host(dev, &gpu->buf_out, out_bytes)) {
        fprintf(stderr, "[LFM-NV ERROR] Failed to allocate host activation buffers.\n");
        lfm_nv_engine_free(gpu);
        return false;
    }

    return true;
}

void lfm_nv_engine_free(LFMNVEngine *gpu) {
    if (!gpu) return;
    LFMNVDevice *dev = &gpu->vk;

    lfm_nv_buffer_free(dev, &gpu->buf_w_shared);
    lfm_nv_buffer_free(dev, &gpu->buf_x);
    lfm_nv_buffer_free(dev, &gpu->buf_out);

    if (gpu->desc_pool)   dev->fn.vkDestroyDescriptorPool(dev->device, gpu->desc_pool, NULL);
    if (gpu->pipe_q4_k)    dev->fn.vkDestroyPipeline(dev->device, gpu->pipe_q4_k, NULL);
    if (gpu->pipe_layout)  dev->fn.vkDestroyPipelineLayout(dev->device, gpu->pipe_layout, NULL);
    if (gpu->desc_layout)  dev->fn.vkDestroyDescriptorSetLayout(dev->device, gpu->desc_layout, NULL);

    lfm_nv_free(dev);
    memset(gpu, 0, sizeof(*gpu));
}

bool lfm_nv_engine_alloc_and_upload_vram(LFMNVEngine *gpu, LFMVulkanBuffer_NV *vram_buf,
                                        const void *w_host, size_t w_bytes) {
    if (!gpu || !vram_buf || !w_host || w_bytes == 0) return false;

    if (!lfm_nv_buffer_alloc_vram(&gpu->vk, vram_buf, w_bytes)) {
        return false;
    }

    return lfm_nv_upload_vram(&gpu->vk, vram_buf, w_host, w_bytes);
}

bool lfm_nv_engine_gemv_q4_k_vram(LFMNVEngine *gpu, float *out, const LFMVulkanBuffer_NV *vram_buf,
                                  const float *x, uint32_t n_cols, uint32_t n_rows) {
    if (!gpu || !out || !vram_buf || !x) return false;

    LFMNVDevice *dev = &gpu->vk;
    size_t x_bytes   = (size_t)n_cols * sizeof(float);
    size_t out_bytes = (size_t)n_rows * sizeof(float);

    if (x_bytes > gpu->max_x_size || out_bytes > gpu->max_out_size) {
        return false;
    }

    /* 1. Copy input vector to mapped host buffer */
    memcpy(gpu->buf_x.host_ptr, x, x_bytes);

    /* 2. Bind VRAM weight buffer and host activation buffers to descriptor set */
    VkDescriptorBufferInfo bInfos[3] = {
        { .buffer = vram_buf->buffer,    .offset = 0, .range = vram_buf->size },
        { .buffer = gpu->buf_x.buffer,   .offset = 0, .range = x_bytes },
        { .buffer = gpu->buf_out.buffer, .offset = 0, .range = out_bytes }
    };

    VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = gpu->desc_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bInfos[0]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = gpu->desc_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bInfos[1]
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = gpu->desc_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bInfos[2]
        }
    };

    dev->fn.vkUpdateDescriptorSets(dev->device, 3, writes, 0, NULL);

    /* 3. Record & Dispatch Compute Command Buffer */
    VkCommandBufferBeginInfo bInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    dev->fn.vkResetCommandBuffer(dev->cmd_buffer, 0);
    dev->fn.vkBeginCommandBuffer(dev->cmd_buffer, &bInfo);

    dev->fn.vkCmdBindPipeline(dev->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->pipe_q4_k);
    dev->fn.vkCmdBindDescriptorSets(dev->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, gpu->pipe_layout, 0, 1, &gpu->desc_set, 0, NULL);

    PushConstantsNV pc = { .n_cols = n_cols, .n_rows = n_rows };
    dev->fn.vkCmdPushConstants(dev->cmd_buffer, gpu->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);

    /* 64 threads per workgroup */
    uint32_t group_count_x = (n_rows + 63) / 64;
    dev->fn.vkCmdDispatch(dev->cmd_buffer, group_count_x, 1, 1);

    dev->fn.vkEndCommandBuffer(dev->cmd_buffer);

    /* 4. Execute on NVIDIA Queue and Wait */
    VkSubmitInfo sInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &dev->cmd_buffer
    };

    VkResult q_res = dev->fn.vkQueueSubmit(dev->queue, 1, &sInfo, NULL);
    if (q_res != VK_SUCCESS) {
        fprintf(stderr, "[LFM-NV ERROR] vkQueueSubmit failed: code %d\n", q_res);
        return false;
    }
    q_res = dev->fn.vkQueueWaitIdle(dev->queue);
    if (q_res != VK_SUCCESS) {
        fprintf(stderr, "[LFM-NV ERROR] vkQueueWaitIdle failed: code %d\n", q_res);
        return false;
    }

    /* 5. Copy result from host buffer */
    memcpy(out, gpu->buf_out.host_ptr, out_bytes);
    return true;
}

bool lfm_nv_engine_gemv_q4_k(LFMNVEngine *gpu, float *out, const void *w_data,
                             const float *x, uint32_t n_cols, uint32_t n_rows) {
    size_t nb = n_cols / 256;
    size_t w_bytes = (size_t)n_rows * nb * 144;

    if (w_bytes > gpu->max_w_size) return false;

    if (!lfm_nv_upload_vram(&gpu->vk, &gpu->buf_w_shared, w_data, w_bytes)) {
        return false;
    }

    return lfm_nv_engine_gemv_q4_k_vram(gpu, out, &gpu->buf_w_shared, x, n_cols, n_rows);
}
