#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

/* Minimal Vulkan types to enumerate physical devices without requiring any SDK headers */
typedef uint32_t VkFlags;
typedef uint32_t VkStructureType;
typedef int32_t  VkResult;
typedef void*    VkInstance;
typedef void*    VkPhysicalDevice;

#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_SUCCESS 0

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
    uint8_t  limits_and_sparse[512]; /* padding for remaining struct fields */
} VkPhysicalDeviceProperties_min;

int main(void) {
    printf("=== Dynamic GPU Detector (Zero-SDK Driver Probe) ===\n");

    /* Probe Vulkan runtime driver (libvulkan.so.1) */
    void *vk = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!vk) {
        printf("[Vulkan] libvulkan.so.1 not found or failed to load: %s\n", dlerror());
    } else {
        printf("[Vulkan] Successfully opened libvulkan.so.1 via dlopen\n");

        VkResult (*vkCreateInstance)(const VkInstanceCreateInfo*, const void*, VkInstance*) =
            (VkResult (*)(const VkInstanceCreateInfo*, const void*, VkInstance*))dlsym(vk, "vkCreateInstance");
        VkResult (*vkEnumeratePhysicalDevices)(VkInstance, uint32_t*, VkPhysicalDevice*) =
            (VkResult (*)(VkInstance, uint32_t*, VkPhysicalDevice*))dlsym(vk, "vkEnumeratePhysicalDevices");
        void (*vkGetPhysicalDeviceProperties)(VkPhysicalDevice, void*) =
            (void (*)(VkPhysicalDevice, void*))dlsym(vk, "vkGetPhysicalDeviceProperties");
        void (*vkDestroyInstance)(VkInstance, const void*) =
            (void (*)(VkInstance, const void*))dlsym(vk, "vkDestroyInstance");

        if (vkCreateInstance && vkEnumeratePhysicalDevices && vkGetPhysicalDeviceProperties && vkDestroyInstance) {
            VkApplicationInfo appInfo = {
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pApplicationName = "LFM2-GPU-Detector",
                .apiVersion = (1 << 22) | (1 << 0) /* Vulkan 1.0 baseline */
            };
            VkInstanceCreateInfo createInfo = {
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pApplicationInfo = &appInfo
            };

            VkInstance instance = NULL;
            VkResult res = vkCreateInstance(&createInfo, NULL, &instance);
            if (res == VK_SUCCESS && instance) {
                uint32_t count = 0;
                vkEnumeratePhysicalDevices(instance, &count, NULL);
                printf("[Vulkan] Detected %u physical device(s):\n", count);

                if (count > 0) {
                    VkPhysicalDevice devices[16];
                    if (count > 16) count = 16;
                    vkEnumeratePhysicalDevices(instance, &count, devices);

                    const char *type_names[] = {
                        "Other", "Integrated GPU (iGPU)", "Discrete GPU (dGPU)", "Virtual GPU", "CPU"
                    };

                    for (uint32_t i = 0; i < count; ++i) {
                        /* VkPhysicalDeviceProperties is ~824 bytes; use 4096 byte aligned buffer to safely hold it */
                        _Alignas(8) uint8_t props_buf[4096];
                        memset(props_buf, 0, sizeof(props_buf));
                        vkGetPhysicalDeviceProperties(devices[i], props_buf);

                        VkPhysicalDeviceProperties_min *props = (VkPhysicalDeviceProperties_min *)props_buf;

                        const char *type_str = (props->deviceType <= 4) ? type_names[props->deviceType] : "Unknown";
                        printf("  -> GPU [%u]: %s\n", i, props->deviceName);
                        printf("     Type     : %s\n", type_str);
                        printf("     VendorID : 0x%04X (", props->vendorID);
                        if (props->vendorID == 0x8086) printf("Intel");
                        else if (props->vendorID == 0x10DE) printf("NVIDIA");
                        else if (props->vendorID == 0x1002) printf("AMD");
                        else printf("Unknown");
                        printf(")\n");
                        printf("     DeviceID : 0x%04X\n", props->deviceID);
                    }
                }
                vkDestroyInstance(instance, NULL);
            } else {
                printf("[Vulkan] vkCreateInstance returned error %d\n", res);
            }
        } else {
            printf("[Vulkan] Failed to resolve required Vulkan function pointers\n");
        }
        dlclose(vk);
    }

    /* Probe OpenCL driver (libOpenCL.so.1) */
    printf("\n[OpenCL] Probing libOpenCL.so.1...\n");
    void *cl = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!cl) {
        printf("[OpenCL] libOpenCL.so.1 not found: %s\n", dlerror());
    } else {
        printf("[OpenCL] Successfully opened libOpenCL.so.1 via dlopen\n");
        dlclose(cl);
    }

    return 0;
}
