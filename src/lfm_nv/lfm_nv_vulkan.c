#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "lfm_nv_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define LOAD_VK_SYM(handle, name) do {                                    \
    dev->fn.name = dlsym(handle, #name);                                  \
    if (!dev->fn.name) {                                                  \
        fprintf(stderr, "[LFM-NV ERROR] Missing symbol: %s\n", #name);    \
        return false;                                                     \
    }                                                                     \
} while (0)

bool lfm_nv_init(LFMNVDevice *dev) {
    if (!dev) return false;
    memset(dev, 0, sizeof(*dev));

    /* 1. Open system Vulkan driver */
    dev->lib_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!dev->lib_handle) {
        dev->lib_handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!dev->lib_handle) {
        fprintf(stderr, "[LFM-NV ERROR] Could not open libvulkan.so.1: %s\n", dlerror());
        return false;
    }

    LOAD_VK_SYM(dev->lib_handle, vkCreateInstance);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyInstance);
    LOAD_VK_SYM(dev->lib_handle, vkEnumeratePhysicalDevices);
    LOAD_VK_SYM(dev->lib_handle, vkGetPhysicalDeviceProperties);
    LOAD_VK_SYM(dev->lib_handle, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK_SYM(dev->lib_handle, vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK_SYM(dev->lib_handle, vkCreateDevice);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyDevice);
    LOAD_VK_SYM(dev->lib_handle, vkGetDeviceQueue);
    LOAD_VK_SYM(dev->lib_handle, vkCreateCommandPool);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyCommandPool);
    LOAD_VK_SYM(dev->lib_handle, vkAllocateCommandBuffers);
    LOAD_VK_SYM(dev->lib_handle, vkFreeCommandBuffers);
    LOAD_VK_SYM(dev->lib_handle, vkBeginCommandBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkEndCommandBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkCreateBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkGetBufferMemoryRequirements);
    LOAD_VK_SYM(dev->lib_handle, vkAllocateMemory);
    LOAD_VK_SYM(dev->lib_handle, vkFreeMemory);
    LOAD_VK_SYM(dev->lib_handle, vkBindBufferMemory);
    LOAD_VK_SYM(dev->lib_handle, vkMapMemory);
    LOAD_VK_SYM(dev->lib_handle, vkUnmapMemory);
    LOAD_VK_SYM(dev->lib_handle, vkCreateShaderModule);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyShaderModule);
    LOAD_VK_SYM(dev->lib_handle, vkCreateDescriptorSetLayout);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyDescriptorSetLayout);
    LOAD_VK_SYM(dev->lib_handle, vkCreatePipelineLayout);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyPipelineLayout);
    LOAD_VK_SYM(dev->lib_handle, vkCreateComputePipelines);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyPipeline);
    LOAD_VK_SYM(dev->lib_handle, vkCreateDescriptorPool);
    LOAD_VK_SYM(dev->lib_handle, vkDestroyDescriptorPool);
    LOAD_VK_SYM(dev->lib_handle, vkAllocateDescriptorSets);
    LOAD_VK_SYM(dev->lib_handle, vkUpdateDescriptorSets);
    LOAD_VK_SYM(dev->lib_handle, vkCmdBindPipeline);
    LOAD_VK_SYM(dev->lib_handle, vkCmdBindDescriptorSets);
    LOAD_VK_SYM(dev->lib_handle, vkCmdPushConstants);
    LOAD_VK_SYM(dev->lib_handle, vkCmdDispatch);
    LOAD_VK_SYM(dev->lib_handle, vkCmdCopyBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkQueueSubmit);
    LOAD_VK_SYM(dev->lib_handle, vkQueueWaitIdle);
    LOAD_VK_SYM(dev->lib_handle, vkResetCommandBuffer);

    /* 2. Create Vulkan Instance */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "LFM-NV",
        .applicationVersion = 1,
        .pEngineName = "LFM-NV-Engine",
        .engineVersion = 1,
        .apiVersion = (1 << 22) | (1 << 12) /* Vulkan 1.1 */
    };

    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo
    };

    if (dev->fn.vkCreateInstance(&instInfo, NULL, &dev->instance) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-NV ERROR] Failed to create Vulkan instance.\n");
        return false;
    }

    /* 3. Specifically Select NVIDIA Discrete GPU (Vendor 0x10DE) */
    uint32_t deviceCount = 0;
    dev->fn.vkEnumeratePhysicalDevices(dev->instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        fprintf(stderr, "[LFM-NV ERROR] No Vulkan physical devices found.\n");
        return false;
    }

    VkPhysicalDevice devices[16];
    if (deviceCount > 16) deviceCount = 16;
    dev->fn.vkEnumeratePhysicalDevices(dev->instance, &deviceCount, devices);

    bool found_nv = false;
    for (uint32_t i = 0; i < deviceCount; ++i) {
        _Alignas(8) uint8_t props_buf[4096];
        memset(props_buf, 0, sizeof(props_buf));
        dev->fn.vkGetPhysicalDeviceProperties(devices[i], props_buf);
        VkPhysicalDeviceProperties_min *p = (VkPhysicalDeviceProperties_min *)props_buf;

        if (p->vendorID == 0x10DE) { /* NVIDIA */
            dev->phys_device = devices[i];
            snprintf(dev->device_name, sizeof(dev->device_name), "%.255s", p->deviceName);
            found_nv = true;
            break;
        }
    }

    if (!found_nv) {
        fprintf(stderr, "[LFM-NV ERROR] No NVIDIA GPU found on system (vendor 0x10DE).\n");
        return false;
    }

    /* 4. Query Memory Heaps for Dedicated GDDR5 VRAM */
    dev->fn.vkGetPhysicalDeviceMemoryProperties(dev->phys_device, &dev->mem_props);

    size_t max_vram = 0;
    for (uint32_t h = 0; h < dev->mem_props.memoryHeapCount; ++h) {
        if (dev->mem_props.memoryHeaps[h].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            if (dev->mem_props.memoryHeaps[h].size > max_vram) {
                max_vram = dev->mem_props.memoryHeaps[h].size;
                dev->vram_heap_idx = h;
                dev->vram_total_bytes = max_vram;
            }
        }
    }

    /* 5. Find Compute Queue Family */
    uint32_t queueFamilyCount = 0;
    dev->fn.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys_device, &queueFamilyCount, NULL);
    if (queueFamilyCount == 0) return false;

    VkQueueFamilyProperties_min qprops[16];
    if (queueFamilyCount > 16) queueFamilyCount = 16;
    dev->fn.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys_device, &queueFamilyCount, qprops);

    bool found_queue = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            dev->compute_queue_family = i;
            found_queue = true;
            break;
        }
    }
    if (!found_queue) return false;

    /* 6. Create Logical Device */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = dev->compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };

    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo
    };

    if (dev->fn.vkCreateDevice(dev->phys_device, &devInfo, NULL, &dev->device) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-NV ERROR] vkCreateDevice failed.\n");
        return false;
    }

    dev->fn.vkGetDeviceQueue(dev->device, dev->compute_queue_family, 0, &dev->queue);

    /* 7. Create Command Pool & Primary Command Buffer */
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = 0x00000002, /* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT */
        .queueFamilyIndex = dev->compute_queue_family
    };

    if (dev->fn.vkCreateCommandPool(dev->device, &poolInfo, NULL, &dev->cmd_pool) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferAllocateInfo cmdInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    if (dev->fn.vkAllocateCommandBuffers(dev->device, &cmdInfo, &dev->cmd_buffer) != VK_SUCCESS) {
        return false;
    }

    return true;
}

void lfm_nv_free(LFMNVDevice *dev) {
    if (!dev || !dev->device) return;
    dev->fn.vkQueueWaitIdle(dev->queue);
    if (dev->cmd_pool) dev->fn.vkDestroyCommandPool(dev->device, dev->cmd_pool, NULL);
    if (dev->device) dev->fn.vkDestroyDevice(dev->device, NULL);
    if (dev->instance) dev->fn.vkDestroyInstance(dev->instance, NULL);
    if (dev->lib_handle) dlclose(dev->lib_handle);
    memset(dev, 0, sizeof(*dev));
}

static uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties *mem_props,
                                 uint32_t type_bits, VkFlags req_props, bool require_device_local_heap) {
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        if (type_bits & (1u << i)) {
            VkFlags props = mem_props->memoryTypes[i].propertyFlags;
            if ((props & req_props) == req_props) {
                if (require_device_local_heap) {
                    uint32_t h = mem_props->memoryTypes[i].heapIndex;
                    if (mem_props->memoryHeaps[h].flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                        return i;
                    }
                } else {
                    return i;
                }
            }
        }
    }
    /* Fallback */
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        if (type_bits & (1u << i)) return i;
    }
    return 0;
}

/* Allocates buffer in direct-mapped GDDR5 VRAM (Type 1 BAR) or host GART (Type 2) */
bool lfm_nv_buffer_alloc_vram(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf, size_t size) {
    if (!dev || !buf || size == 0) return false;
    memset(buf, 0, sizeof(*buf));
    buf->size = size;

    VkBufferCreateInfo bInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    if (dev->fn.vkCreateBuffer(dev->device, &bInfo, NULL, &buf->buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memReqs;
    dev->fn.vkGetBufferMemoryRequirements(dev->device, buf->buffer, &memReqs);

    /* Prefer Type 1 (DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT in Heap 1) */
    uint32_t mem_type = find_memory_type(&dev->mem_props, memReqs.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);

    buf->is_device_vram = (mem_type == 1);

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = mem_type
    };

    if (dev->fn.vkAllocateMemory(dev->device, &allocInfo, NULL, &buf->memory) != VK_SUCCESS) {
        /* Fallback to Type 2 if Type 1 is exhausted */
        mem_type = find_memory_type(&dev->mem_props, memReqs.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false);
        allocInfo.memoryTypeIndex = mem_type;
        buf->is_device_vram = false;
        if (dev->fn.vkAllocateMemory(dev->device, &allocInfo, NULL, &buf->memory) != VK_SUCCESS) {
            dev->fn.vkDestroyBuffer(dev->device, buf->buffer, NULL);
            return false;
        }
    }

    dev->fn.vkBindBufferMemory(dev->device, buf->buffer, buf->memory, 0);

    if (dev->fn.vkMapMemory(dev->device, buf->memory, 0, size, 0, &buf->host_ptr) != VK_SUCCESS) {
        lfm_nv_buffer_free(dev, buf);
        return false;
    }
    return true;
}

/* Allocates host-visible buffer for activations */
bool lfm_nv_buffer_alloc_host(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf, size_t size) {
    return lfm_nv_buffer_alloc_vram(dev, buf, size);
}

/* Copies data from host into mapped GPU buffer directly without staging copy */
bool lfm_nv_upload_vram(LFMNVDevice *dev, LFMVulkanBuffer_NV *dst_vram, const void *src_host, size_t size) {
    (void)dev;
    if (!dst_vram || !dst_vram->host_ptr || !src_host || size == 0) return false;
    memcpy(dst_vram->host_ptr, src_host, size);
    return true;
}

void lfm_nv_buffer_free(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf) {
    if (!dev || !buf || !buf->buffer) return;
    if (buf->host_ptr) {
        dev->fn.vkUnmapMemory(dev->device, buf->memory);
        buf->host_ptr = NULL;
    }
    if (buf->memory) {
        dev->fn.vkFreeMemory(dev->device, buf->memory, NULL);
        buf->memory = NULL;
    }
    if (buf->buffer) {
        dev->fn.vkDestroyBuffer(dev->device, buf->buffer, NULL);
        buf->buffer = NULL;
    }
}
