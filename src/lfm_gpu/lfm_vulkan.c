#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "lfm_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define LOAD_VK_SYM(handle, name) do {                                    \
    dev->fn.name = dlsym(handle, #name);                                  \
    if (!dev->fn.name) {                                                  \
        fprintf(stderr, "[LFM-VULKAN ERROR] Missing symbol: %s\n", #name); \
        return false;                                                     \
    }                                                                     \
} while (0)

bool lfm_vulkan_init(LFMVulkanDevice *dev, int target_device_index) {
    if (!dev) return false;
    memset(dev, 0, sizeof(*dev));

    /* 1. Open system Vulkan driver */
    dev->lib_handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!dev->lib_handle) {
        dev->lib_handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!dev->lib_handle) {
        fprintf(stderr, "[LFM-VULKAN ERROR] Could not load libvulkan.so.1: %s\n", dlerror());
        return false;
    }

    /* 2. Resolve Instance-level entry points */
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
    LOAD_VK_SYM(dev->lib_handle, vkBeginCommandBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkEndCommandBuffer);
    LOAD_VK_SYM(dev->lib_handle, vkCmdBindPipeline);
    LOAD_VK_SYM(dev->lib_handle, vkCmdBindDescriptorSets);
    LOAD_VK_SYM(dev->lib_handle, vkCmdPushConstants);
    LOAD_VK_SYM(dev->lib_handle, vkCmdDispatch);
    LOAD_VK_SYM(dev->lib_handle, vkQueueSubmit);
    LOAD_VK_SYM(dev->lib_handle, vkQueueWaitIdle);

    /* 3. Create Vulkan Instance */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "LFM-GPU",
        .applicationVersion = 1,
        .pEngineName = "LFM-Inference",
        .engineVersion = 1,
        .apiVersion = (1 << 22) | (1 << 0) /* Vulkan 1.0 baseline */
    };

    VkInstanceCreateInfo instInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo
    };

    if (dev->fn.vkCreateInstance(&instInfo, NULL, &dev->instance) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-VULKAN ERROR] vkCreateInstance failed.\n");
        return false;
    }

    /* 4. Enumerate Physical Devices */
    uint32_t deviceCount = 0;
    dev->fn.vkEnumeratePhysicalDevices(dev->instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        fprintf(stderr, "[LFM-VULKAN ERROR] No Vulkan physical devices found.\n");
        return false;
    }

    VkPhysicalDevice devices[16];
    if (deviceCount > 16) deviceCount = 16;
    dev->fn.vkEnumeratePhysicalDevices(dev->instance, &deviceCount, devices);

    uint32_t chosen_idx = 0;

    for (uint32_t i = 0; i < deviceCount; ++i) {
        _Alignas(8) uint8_t props_buf[4096];
        memset(props_buf, 0, sizeof(props_buf));
        dev->fn.vkGetPhysicalDeviceProperties(devices[i], props_buf);
        VkPhysicalDeviceProperties_min *p = (VkPhysicalDeviceProperties_min *)props_buf;

        if (target_device_index >= 0 && (uint32_t)target_device_index == i) {
            chosen_idx = i;
            break;
        }

        /* Default preference: Intel Integrated GPU (0x8086) */
        if (target_device_index < 0 && p->vendorID == 0x8086) {
            chosen_idx = i;
            break;
        }
    }

    dev->phys_device = devices[chosen_idx];

    _Alignas(8) uint8_t chosen_props_buf[4096];
    memset(chosen_props_buf, 0, sizeof(chosen_props_buf));
    dev->fn.vkGetPhysicalDeviceProperties(dev->phys_device, chosen_props_buf);
    VkPhysicalDeviceProperties_min *p = (VkPhysicalDeviceProperties_min *)chosen_props_buf;

    snprintf(dev->device_name, sizeof(dev->device_name), "%.255s", p->deviceName);
    dev->vendor_id = p->vendorID;
    dev->is_integrated = (p->deviceType == 1); /* 1 = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU */

    dev->fn.vkGetPhysicalDeviceMemoryProperties(dev->phys_device, &dev->mem_props);

    /* 5. Find Compute Queue Family */
    uint32_t queueFamilyCount = 0;
    dev->fn.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys_device, &queueFamilyCount, NULL);
    if (queueFamilyCount == 0) {
        fprintf(stderr, "[LFM-VULKAN ERROR] No queue families found.\n");
        return false;
    }

    VkQueueFamilyProperties qprops[16];
    if (queueFamilyCount > 16) queueFamilyCount = 16;
    dev->fn.vkGetPhysicalDeviceQueueFamilyProperties(dev->phys_device, &queueFamilyCount, qprops);

    bool found_compute_queue = false;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            dev->compute_queue_family = i;
            found_compute_queue = true;
            break;
        }
    }

    if (!found_compute_queue) {
        fprintf(stderr, "[LFM-VULKAN ERROR] No compute-capable queue family found on %s.\n", dev->device_name);
        return false;
    }

    /* 6. Create Logical Device */
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = dev->compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority
    };

    VkDeviceCreateInfo devInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo
    };

    if (dev->fn.vkCreateDevice(dev->phys_device, &devInfo, NULL, &dev->device) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-VULKAN ERROR] vkCreateDevice failed.\n");
        return false;
    }

    dev->fn.vkGetDeviceQueue(dev->device, dev->compute_queue_family, 0, &dev->queue);

    /* 7. Create Command Pool & Buffer */
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = 0x00000002, /* VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT */
        .queueFamilyIndex = dev->compute_queue_family
    };

    if (dev->fn.vkCreateCommandPool(dev->device, &poolInfo, NULL, &dev->cmd_pool) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-VULKAN ERROR] vkCreateCommandPool failed.\n");
        return false;
    }

    VkCommandBufferAllocateInfo cmdInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = dev->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    if (dev->fn.vkAllocateCommandBuffers(dev->device, &cmdInfo, &dev->cmd_buffer) != VK_SUCCESS) {
        fprintf(stderr, "[LFM-VULKAN ERROR] vkAllocateCommandBuffers failed.\n");
        return false;
    }

    return true;
}

static uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties *mem_props,
                                 uint32_t type_bits, VkFlags req_props) {
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mem_props->memoryTypes[i].propertyFlags & req_props) == req_props) {
            return i;
        }
    }
    /* Fallback: any matching type */
    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        if (type_bits & (1u << i)) return i;
    }
    return 0;
}

bool lfm_vulkan_buffer_alloc(LFMVulkanDevice *dev, LFMVulkanBuffer *buf, size_t size, bool host_visible) {
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

    VkFlags req_flags = host_visible ?
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) :
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    uint32_t mem_type = find_memory_type(&dev->mem_props, memReqs.memoryTypeBits, req_flags);

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReqs.size,
        .memoryTypeIndex = mem_type
    };

    if (dev->fn.vkAllocateMemory(dev->device, &allocInfo, NULL, &buf->memory) != VK_SUCCESS) {
        dev->fn.vkDestroyBuffer(dev->device, buf->buffer, NULL);
        return false;
    }

    dev->fn.vkBindBufferMemory(dev->device, buf->buffer, buf->memory, 0);

    if (host_visible) {
        if (dev->fn.vkMapMemory(dev->device, buf->memory, 0, size, 0, &buf->host_ptr) != VK_SUCCESS) {
            lfm_vulkan_buffer_free(dev, buf);
            return false;
        }
    }

    return true;
}

void lfm_vulkan_buffer_free(LFMVulkanDevice *dev, LFMVulkanBuffer *buf) {
    if (!dev || !buf) return;
    if (buf->host_ptr) {
        dev->fn.vkUnmapMemory(dev->device, buf->memory);
        buf->host_ptr = NULL;
    }
    if (buf->buffer) {
        dev->fn.vkDestroyBuffer(dev->device, buf->buffer, NULL);
        buf->buffer = 0;
    }
    if (buf->memory) {
        dev->fn.vkFreeMemory(dev->device, buf->memory, NULL);
        buf->memory = 0;
    }
    buf->size = 0;
}

void lfm_vulkan_free(LFMVulkanDevice *dev) {
    if (!dev) return;
    if (dev->device) {
        dev->fn.vkQueueWaitIdle(dev->queue);
        if (dev->cmd_buffer && dev->cmd_pool) {
            dev->fn.vkFreeCommandBuffers(dev->device, dev->cmd_pool, 1, &dev->cmd_buffer);
        }
        if (dev->cmd_pool) {
            dev->fn.vkDestroyCommandPool(dev->device, dev->cmd_pool, NULL);
        }
        dev->fn.vkDestroyDevice(dev->device, NULL);
    }
    if (dev->instance) {
        dev->fn.vkDestroyInstance(dev->instance, NULL);
    }
    if (dev->lib_handle) {
        dlclose(dev->lib_handle);
    }
    memset(dev, 0, sizeof(*dev));
}
