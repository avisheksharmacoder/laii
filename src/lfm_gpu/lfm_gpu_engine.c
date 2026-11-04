#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "lfm_gpu_engine.h"
#include "gemv_f32_spv.h"
#include "gemv_q4_k_spv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t n_cols;
    uint32_t n_rows;
} PushConstants;

static VkPipeline create_compute_pipeline(LFMVulkanDevice *dev, VkPipelineLayout layout,
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
            .sType = (VkStructureType)18 /* VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO */,
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

bool lfm_gpu_engine_init(LFMGPUEngine *gpu, int target_device_index, size_t max_matrix_bytes) {
    if (!gpu) return false;
    memset(gpu, 0, sizeof(*gpu));

    if (!lfm_vulkan_init(&gpu->vk, target_device_index)) {
        return false;
    }

    LFMVulkanDevice *dev = &gpu->vk;

    /* 1. Create Descriptor Set Layout (Bindings 0: W, 1: X, 2: Out) */
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
        lfm_gpu_engine_free(gpu);
        return false;
    }

    /* 2. Create Pipeline Layout with Push Constants */
    VkPushConstantRange pcRange = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants)
    };

    VkPipelineLayoutCreateInfo plInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &gpu->desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange
    };

    if (dev->fn.vkCreatePipelineLayout(dev->device, &plInfo, NULL, &gpu->pipe_layout) != VK_SUCCESS) {
        lfm_gpu_engine_free(gpu);
        return false;
    }

    /* 3. Create Compute Pipelines from Embedded SPIR-V */
    gpu->pipe_gemv_f32 = create_compute_pipeline(dev, gpu->pipe_layout,
        src_lfm_gpu_shaders_gemv_f32_spv, src_lfm_gpu_shaders_gemv_f32_spv_len);

    gpu->pipe_gemv_q4_k = create_compute_pipeline(dev, gpu->pipe_layout,
        src_lfm_gpu_shaders_gemv_q4_k_spv, src_lfm_gpu_shaders_gemv_q4_k_spv_len);

    if (!gpu->pipe_gemv_f32 || !gpu->pipe_gemv_q4_k) {
        fprintf(stderr, "[LFM-GPU ERROR] Failed to create compute pipelines.\n");
        lfm_gpu_engine_free(gpu);
        return false;
    }

    /* 4. Create Descriptor Pool & Allocate Set */
    VkDescriptorPoolSize poolSizes[1] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 3 }
    };
    VkDescriptorPoolCreateInfo dpInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = poolSizes
    };

    if (dev->fn.vkCreateDescriptorPool(dev->device, &dpInfo, NULL, &gpu->desc_pool) != VK_SUCCESS) {
        lfm_gpu_engine_free(gpu);
        return false;
    }

    VkDescriptorSetAllocateInfo dsAlloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = gpu->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &gpu->desc_layout
    };

    if (dev->fn.vkAllocateDescriptorSets(dev->device, &dsAlloc, &gpu->desc_set) != VK_SUCCESS) {
        lfm_gpu_engine_free(gpu);
        return false;
    }

    /* 5. Allocate Host-Coherent Zero-Copy UMA Buffers */
    size_t w_bytes = (max_matrix_bytes > 0) ? max_matrix_bytes : (64 * 1024 * 1024); /* 64 MB default */
    size_t x_bytes = 16 * 1024 * sizeof(float);   /* 64 KB */
    size_t out_bytes = 65536 * sizeof(float);     /* 256 KB (supports vocab projection) */

    gpu->max_w_size   = w_bytes;
    gpu->max_x_size   = x_bytes;
    gpu->max_out_size = out_bytes;

    if (!lfm_vulkan_buffer_alloc(dev, &gpu->buf_w, w_bytes, true) ||
        !lfm_vulkan_buffer_alloc(dev, &gpu->buf_x, x_bytes, true) ||
        !lfm_vulkan_buffer_alloc(dev, &gpu->buf_out, out_bytes, true)) {
        fprintf(stderr, "[LFM-GPU ERROR] Failed to allocate GPU buffers.\n");
        lfm_gpu_engine_free(gpu);
        return false;
    }

    /* 6. Bind Buffers to Descriptor Set */
    VkDescriptorBufferInfo bInfos[3] = {
        { .buffer = gpu->buf_w.buffer,   .offset = 0, .range = w_bytes },
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
    return true;
}

static bool run_gemv_pipeline(LFMGPUEngine *gpu, VkPipeline pipeline,
                              uint32_t n_cols, uint32_t n_rows) {
    LFMVulkanDevice *dev = &gpu->vk;

    VkCommandBufferBeginInfo bInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    dev->fn.vkBeginCommandBuffer(dev->cmd_buffer, &bInfo);
    dev->fn.vkCmdBindPipeline(dev->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    dev->fn.vkCmdBindDescriptorSets(dev->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    gpu->pipe_layout, 0, 1, &gpu->desc_set, 0, NULL);

    PushConstants pc = { .n_cols = n_cols, .n_rows = n_rows };
    dev->fn.vkCmdPushConstants(dev->cmd_buffer, gpu->pipe_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    /* 64 invocations per workgroup */
    uint32_t workgroups = (n_rows + 63) / 64;
    dev->fn.vkCmdDispatch(dev->cmd_buffer, workgroups, 1, 1);
    dev->fn.vkEndCommandBuffer(dev->cmd_buffer);

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &dev->cmd_buffer
    };

    if (dev->fn.vkQueueSubmit(dev->queue, 1, &submitInfo, 0) != VK_SUCCESS) {
        return false;
    }

    return (dev->fn.vkQueueWaitIdle(dev->queue) == VK_SUCCESS);
}

bool lfm_gpu_gemv_f32(LFMGPUEngine *gpu, float *out, const float *W, const float *x,
                      uint32_t n_cols, uint32_t n_rows) {
    size_t w_bytes = (size_t)n_rows * n_cols * sizeof(float);
    size_t x_bytes = (size_t)n_cols * sizeof(float);
    size_t out_bytes = (size_t)n_rows * sizeof(float);

    if (w_bytes > gpu->max_w_size || x_bytes > gpu->max_x_size || out_bytes > gpu->max_out_size) {
        return false;
    }

    /* Zero-copy write directly to mapped host-coherent GPU buffer */
    memcpy(gpu->buf_w.host_ptr, W, w_bytes);
    memcpy(gpu->buf_x.host_ptr, x, x_bytes);

    if (!run_gemv_pipeline(gpu, gpu->pipe_gemv_f32, n_cols, n_rows)) {
        return false;
    }

    memcpy(out, gpu->buf_out.host_ptr, out_bytes);
    return true;
}

bool lfm_gpu_gemv_q4_k(LFMGPUEngine *gpu, float *out, const void *w_q4_data, const float *x,
                       uint32_t n_cols, uint32_t n_rows) {
    size_t nb = (size_t)n_cols / 256;
    size_t w_bytes = (size_t)n_rows * nb * 144; /* 144 bytes per Q4_K block */
    size_t x_bytes = (size_t)n_cols * sizeof(float);
    size_t out_bytes = (size_t)n_rows * sizeof(float);

    if (w_bytes > gpu->max_w_size || x_bytes > gpu->max_x_size || out_bytes > gpu->max_out_size) {
        return false;
    }

    /* Zero-copy write directly to mapped host-coherent GPU buffer */
    memcpy(gpu->buf_w.host_ptr, w_q4_data, w_bytes);
    memcpy(gpu->buf_x.host_ptr, x, x_bytes);

    if (!run_gemv_pipeline(gpu, gpu->pipe_gemv_q4_k, n_cols, n_rows)) {
        return false;
    }

    memcpy(out, gpu->buf_out.host_ptr, out_bytes);
    return true;
}

void lfm_gpu_engine_free(LFMGPUEngine *gpu) {
    if (!gpu) return;
    LFMVulkanDevice *dev = &gpu->vk;

    if (dev->device) {
        dev->fn.vkQueueWaitIdle(dev->queue);
        lfm_vulkan_buffer_free(dev, &gpu->buf_w);
        lfm_vulkan_buffer_free(dev, &gpu->buf_x);
        lfm_vulkan_buffer_free(dev, &gpu->buf_out);

        if (gpu->desc_pool) {
            dev->fn.vkDestroyDescriptorPool(dev->device, gpu->desc_pool, NULL);
        }
        if (gpu->desc_layout) {
            dev->fn.vkDestroyDescriptorSetLayout(dev->device, gpu->desc_layout, NULL);
        }
        if (gpu->pipe_gemv_f32) {
            dev->fn.vkDestroyPipeline(dev->device, gpu->pipe_gemv_f32, NULL);
        }
        if (gpu->pipe_gemv_q4_k) {
            dev->fn.vkDestroyPipeline(dev->device, gpu->pipe_gemv_q4_k, NULL);
        }
        if (gpu->pipe_layout) {
            dev->fn.vkDestroyPipelineLayout(dev->device, gpu->pipe_layout, NULL);
        }
    }

    lfm_vulkan_free(dev);
    memset(gpu, 0, sizeof(*gpu));
}
