#ifndef LFM_NV_VULKAN_H
#define LFM_NV_VULKAN_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t VkFlags;
typedef uint32_t VkStructureType;
typedef int32_t  VkResult;
typedef uint64_t VkDeviceSize;
typedef void*    VkInstance;
typedef void*    VkPhysicalDevice;
typedef void*    VkDevice;
typedef void*    VkQueue;
typedef void*    VkCommandPool;
typedef void*    VkCommandBuffer;
typedef void*    VkDescriptorSetLayout;
typedef void*    VkDescriptorPool;
typedef void*    VkDescriptorSet;
typedef void*    VkPipelineLayout;
typedef void*    VkPipeline;
typedef void*    VkShaderModule;
typedef void*    VkBuffer;
typedef void*    VkDeviceMemory;

#define VK_SUCCESS 0
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_SUBMIT_INFO 4
#define VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO 39
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO 40
#define VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO 42
#define VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO 12
#define VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO 5
#define VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO 16
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18
#define VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO 33
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO 34
#define VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET 35

#define VK_QUEUE_COMPUTE_BIT 0x00000002
#define VK_QUEUE_TRANSFER_BIT 0x00000004
#define VK_PIPELINE_STAGE_COMPUTE_BIT 0x00000800
#define VK_PIPELINE_STAGE_TRANSFER_BIT 0x00001000
#define VK_SHADER_STAGE_COMPUTE_BIT 0x00000020
#define VK_DESCRIPTOR_TYPE_STORAGE_BUFFER 7
#define VK_PIPELINE_BIND_POINT_GRAPHICS 0
#define VK_PIPELINE_BIND_POINT_COMPUTE 1

#define VK_BUFFER_USAGE_TRANSFER_SRC_BIT 0x00000001
#define VK_BUFFER_USAGE_TRANSFER_DST_BIT 0x00000002
#define VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 0x00000020
#define VK_SHARING_MODE_EXCLUSIVE 0

#define VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 0x00000001
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 0x00000002
#define VK_MEMORY_PROPERTY_HOST_COHERENT_BIT 0x00000004
#define VK_MEMORY_PROPERTY_HOST_CACHED_BIT 0x00000008

#define VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT 0x00000001
#define VK_COMMAND_BUFFER_LEVEL_PRIMARY 0

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    const char*     pApplicationName;
    uint32_t        applicationVersion;
    const char*     pEngineName;
    uint32_t        engineVersion;
    uint32_t        apiVersion;
} VkApplicationInfo;

typedef struct {
    VkStructureType          sType;
    const void*              pNext;
    VkFlags                  flags;
    const VkApplicationInfo* pApplicationInfo;
    uint32_t                 enabledLayerCount;
    const char* const*       ppEnabledLayerNames;
    uint32_t                 enabledExtensionCount;
    const char* const*       ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
    uint32_t apiVersion;
    uint32_t driverVersion;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t deviceType;
    char     deviceName[256];
    uint8_t  pipelineCacheUUID[16];
} VkPhysicalDeviceProperties_min;

typedef struct {
    VkFlags  queueFlags;
    uint32_t queueCount;
    uint32_t timestampValidBits;
    uint32_t minImageTransferGranularity[3];
} VkQueueFamilyProperties_min;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        queueFamilyIndex;
    uint32_t        queueCount;
    const float*    pQueuePriorities;
} VkDeviceQueueCreateInfo;

typedef struct {
    VkStructureType                sType;
    const void*                    pNext;
    VkFlags                        flags;
    uint32_t                       queueCreateInfoCount;
    const VkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t                       enabledLayerCount;
    const char* const*             ppEnabledLayerNames;
    uint32_t                       enabledExtensionCount;
    const char* const*             ppEnabledExtensionNames;
    const void*                    pEnabledFeatures;
} VkDeviceCreateInfo;

typedef struct {
    VkDeviceSize size;
    VkDeviceSize alignment;
    uint32_t     memoryTypeBits;
} VkMemoryRequirements;

typedef struct {
    VkFlags  propertyFlags;
    uint32_t heapIndex;
} VkMemoryType;

typedef struct {
    VkDeviceSize size;
    VkFlags      flags;
} VkMemoryHeap;

typedef struct {
    uint32_t     memoryTypeCount;
    VkMemoryType memoryTypes[32];
    uint32_t     memoryHeapCount;
    VkMemoryHeap memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkDeviceSize    allocationSize;
    uint32_t        memoryTypeIndex;
} VkMemoryAllocateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    VkDeviceSize    size;
    VkFlags         usage;
    uint32_t        sharingMode;
    uint32_t        queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
} VkBufferCreateInfo;

typedef struct {
    VkDeviceSize srcOffset;
    VkDeviceSize dstOffset;
    VkDeviceSize size;
} VkBufferCopy;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        queueFamilyIndex;
} VkCommandPoolCreateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkCommandPool   commandPool;
    uint32_t        level;
    uint32_t        commandBufferCount;
} VkCommandBufferAllocateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    const void*     pInheritanceInfo;
} VkCommandBufferBeginInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    uint32_t        waitSemaphoreCount;
    const void*     pWaitSemaphores;
    const VkFlags*  pWaitDstStageMask;
    uint32_t        commandBufferCount;
    const VkCommandBuffer* pCommandBuffers;
    uint32_t        signalSemaphoreCount;
    const void*     pSignalSemaphores;
} VkSubmitInfo;

typedef struct {
    uint32_t binding;
    uint32_t descriptorType;
    uint32_t descriptorCount;
    VkFlags  stageFlags;
    const void* pImmutableSamplers;
} VkDescriptorSetLayoutBinding;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        bindingCount;
    const VkDescriptorSetLayoutBinding* pBindings;
} VkDescriptorSetLayoutCreateInfo;

typedef struct {
    VkFlags  stageFlags;
    uint32_t offset;
    uint32_t size;
} VkPushConstantRange;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        setLayoutCount;
    const VkDescriptorSetLayout* pSetLayouts;
    uint32_t        pushConstantRangeCount;
    const VkPushConstantRange*   pPushConstantRanges;
} VkPipelineLayoutCreateInfo;

typedef struct {
    uint32_t type;
    uint32_t descriptorCount;
} VkDescriptorPoolSize;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        maxSets;
    uint32_t        poolSizeCount;
    const VkDescriptorPoolSize* pPoolSizes;
} VkDescriptorPoolCreateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkDescriptorPool descriptorPool;
    uint32_t        descriptorSetCount;
    const VkDescriptorSetLayout* pSetLayouts;
} VkDescriptorSetAllocateInfo;

typedef struct {
    VkBuffer     buffer;
    VkDeviceSize offset;
    VkDeviceSize range;
} VkDescriptorBufferInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkDescriptorSet dstSet;
    uint32_t        dstBinding;
    uint32_t        dstArrayElement;
    uint32_t        descriptorCount;
    uint32_t        descriptorType;
    const void*     pImageInfo;
    const VkDescriptorBufferInfo* pBufferInfo;
    const void*     pTexelBufferView;
} VkWriteDescriptorSet;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    size_t          codeSize;
    const uint32_t* pCode;
} VkShaderModuleCreateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    uint32_t        stage;
    VkShaderModule  module;
    const char*     pName;
    const void*     pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;

typedef struct {
    VkStructureType sType;
    const void*     pNext;
    VkFlags         flags;
    VkPipelineShaderStageCreateInfo stage;
    VkPipelineLayout layout;
    VkPipeline       basePipelineHandle;
    int32_t          basePipelineIndex;
} VkComputePipelineCreateInfo;

/* Function Pointer VTable */
typedef struct {
    VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*);
    void     (*vkDestroyInstance)(VkInstance, const void*);
    VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*);
    void     (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, void*);
    void     (*vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice, uint32_t*, void*);
    void     (*vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties*);
    VkResult (*vkCreateDevice)(VkPhysicalDevice, const VkDeviceCreateInfo*, const void*, VkDevice*);
    void     (*vkDestroyDevice)(VkDevice, const void*);
    void     (*vkGetDeviceQueue)(VkDevice, uint32_t, uint32_t, VkQueue*);
    VkResult (*vkCreateCommandPool)(VkDevice, const VkCommandPoolCreateInfo*, const void*, VkCommandPool*);
    void     (*vkDestroyCommandPool)(VkDevice, VkCommandPool, const void*);
    VkResult (*vkAllocateCommandBuffers)(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer*);
    void     (*vkFreeCommandBuffers)(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*);
    VkResult (*vkBeginCommandBuffer)(VkCommandBuffer, const VkCommandBufferBeginInfo*);
    VkResult (*vkEndCommandBuffer)(VkCommandBuffer);
    VkResult (*vkCreateBuffer)(VkDevice, const VkBufferCreateInfo*, const void*, VkBuffer*);
    void     (*vkDestroyBuffer)(VkDevice, VkBuffer, const void*);
    void     (*vkGetBufferMemoryRequirements)(VkDevice, VkBuffer, VkMemoryRequirements*);
    VkResult (*vkAllocateMemory)(VkDevice, const VkMemoryAllocateInfo*, const void*, VkDeviceMemory*);
    void     (*vkFreeMemory)(VkDevice, VkDeviceMemory, const void*);
    VkResult (*vkBindBufferMemory)(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
    VkResult (*vkMapMemory)(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkFlags, void**);
    void     (*vkUnmapMemory)(VkDevice, VkDeviceMemory);
    VkResult (*vkCreateShaderModule)(VkDevice, const VkShaderModuleCreateInfo*, const void*, VkShaderModule*);
    void     (*vkDestroyShaderModule)(VkDevice, VkShaderModule, const void*);
    VkResult (*vkCreateDescriptorSetLayout)(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const void*, VkDescriptorSetLayout*);
    void     (*vkDestroyDescriptorSetLayout)(VkDevice, VkDescriptorSetLayout, const void*);
    VkResult (*vkCreatePipelineLayout)(VkDevice, const VkPipelineLayoutCreateInfo*, const void*, VkPipelineLayout*);
    void     (*vkDestroyPipelineLayout)(VkDevice, VkPipelineLayout, const void*);
    VkResult (*vkCreateComputePipelines)(VkDevice, void*, uint32_t, const VkComputePipelineCreateInfo*, const void*, VkPipeline*);
    void     (*vkDestroyPipeline)(VkDevice, VkPipeline, const void*);
    VkResult (*vkCreateDescriptorPool)(VkDevice, const VkDescriptorPoolCreateInfo*, const void*, VkDescriptorPool*);
    void     (*vkDestroyDescriptorPool)(VkDevice, VkDescriptorPool, const void*);
    VkResult (*vkAllocateDescriptorSets)(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet*);
    void     (*vkUpdateDescriptorSets)(VkDevice, uint32_t, const VkWriteDescriptorSet*, uint32_t, const void*);
    void     (*vkCmdBindPipeline)(VkCommandBuffer, uint32_t, VkPipeline);
    void     (*vkCmdBindDescriptorSets)(VkCommandBuffer, uint32_t, VkPipelineLayout, uint32_t, uint32_t, const VkDescriptorSet*, uint32_t, const uint32_t*);
    void     (*vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout, VkFlags, uint32_t, uint32_t, const void*);
    void     (*vkCmdDispatch)(VkCommandBuffer, uint32_t, uint32_t, uint32_t);
    void     (*vkCmdCopyBuffer)(VkCommandBuffer, VkBuffer, VkBuffer, uint32_t, const VkBufferCopy*);
    VkResult (*vkQueueSubmit)(VkQueue, uint32_t, const VkSubmitInfo*, void*);
    VkResult (*vkQueueWaitIdle)(VkQueue);
    VkResult (*vkResetCommandBuffer)(VkCommandBuffer, VkFlags);
} LFMVulkanVTable_NV;

typedef struct {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    void*          host_ptr;      /* Non-null if host-mapped (Type 1 or Type 2) */
    size_t         size;
    bool           is_device_vram;/* True if allocated in dedicated 2GB GDDR5 VRAM */
} LFMVulkanBuffer_NV;

typedef struct {
    void*                            lib_handle;
    VkInstance                       instance;
    VkPhysicalDevice                 phys_device;
    VkDevice                         device;
    VkQueue                          queue;
    uint32_t                         compute_queue_family;
    VkCommandPool                    cmd_pool;
    VkCommandBuffer                  cmd_buffer;
    VkPhysicalDeviceMemoryProperties mem_props;
    LFMVulkanVTable_NV               fn;
    char                             device_name[256];
    uint32_t                         vram_heap_idx;
    size_t                           vram_total_bytes;
} LFMNVDevice;

/* Initializes Vulkan and specifically selects the NVIDIA GPU */
bool lfm_nv_init(LFMNVDevice *dev);
void lfm_nv_free(LFMNVDevice *dev);

/* Allocates a buffer in dedicated GDDR5 VRAM (Heap 0) */
bool lfm_nv_buffer_alloc_vram(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf, size_t size);

/* Allocates a host-visible buffer for transfers or activations */
bool lfm_nv_buffer_alloc_host(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf, size_t size);

/* Uploads data from host pointer into dedicated GDDR5 VRAM via staging copy */
bool lfm_nv_upload_vram(LFMNVDevice *dev, LFMVulkanBuffer_NV *dst_vram, const void *src_host, size_t size);

/* Frees buffer and unmaps memory */
void lfm_nv_buffer_free(LFMNVDevice *dev, LFMVulkanBuffer_NV *buf);

#ifdef __cplusplus
}
#endif

#endif /* LFM_NV_VULKAN_H */
