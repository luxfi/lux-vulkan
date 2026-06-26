// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Vulkan compute backend plugin for lux-accel.
//
// Targets core Vulkan 1.1 compute. Kernels are SPIR-V: create_kernel_from_bundle
// consumes a SPIR-V binary directly (vkCreateShaderModule); create_kernel_from_source
// compiles GLSL compute -> SPIR-V at runtime via libshaderc when LUX_HAS_SHADERC is
// set (the same "compile from source" path CUDA gets from NVRTC and Metal from MSL).
//
// Portability: on macOS this runs through MoltenVK (the VK_KHR_portability_*
// extensions are negotiated automatically when present), so the plugin executes
// real compute on Apple GPUs; on Linux it drives native AMD/NVIDIA/Intel ICDs
// unchanged. When the Vulkan loader/headers are absent at configure time, the
// plugin compiles as a stub that reports is_available()==0.

#include <lux/accel/backend_api.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(LUX_HAS_VULKAN) && LUX_HAS_VULKAN
#include <vulkan/vulkan.h>
#if defined(LUX_HAS_SHADERC) && LUX_HAS_SHADERC
#include <shaderc/shaderc.hpp>
#endif

#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif
#endif

namespace {

const lux_core_api_t* g_core_api = nullptr;

inline void log_error(const char* msg) {
    if (g_core_api && g_core_api->log_error) g_core_api->log_error(msg);
}
inline void log_info(const char* msg) {
    if (g_core_api && g_core_api->log_info) g_core_api->log_info(msg);
}

#if defined(LUX_HAS_VULKAN) && LUX_HAS_VULKAN

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

VkInstance g_instance = VK_NULL_HANDLE;
std::vector<VkPhysicalDevice> g_physical_devices;

struct VulkanDevice {
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t compute_family = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    std::string name;
};

struct VulkanBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint32_t usage = 0;
    void* mapped = nullptr;
};

struct VulkanKernel {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    std::string entry;
    uint32_t wg_x = 256, wg_y = 1, wg_z = 1;  // informational; real size is in the SPIR-V

    // index -> (buffer, offset, range) for storage buffers
    std::map<uint32_t, std::tuple<VkBuffer, VkDeviceSize, VkDeviceSize>> buffer_args;
    // index -> raw bytes, bound as a uniform buffer
    std::map<uint32_t, std::vector<uint8_t>> byte_args;

    // Pipeline cache keyed by the binding signature (stable across dispatches).
    std::string cached_sig;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

struct VulkanQueue {
    VulkanDevice* dev = nullptr;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool recording = false;
    bool submitted = false;
    // Resources owned by the in-flight submission; freed after the fence signals.
    std::vector<VkBuffer> ephemeral_buffers;
    std::vector<VkDeviceMemory> ephemeral_memory;
    std::vector<VkDescriptorPool> ephemeral_pools;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties& mp,
                          uint32_t type_bits,
                          VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred) {
    // First pass: required + preferred. Second pass: required only.
    for (int pass = 0; pass < 2; ++pass) {
        VkMemoryPropertyFlags want = required | (pass == 0 ? preferred : 0);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
    }
    return UINT32_MAX;
}

const char* vendor_name(uint32_t vendor_id) {
    switch (vendor_id) {
        case 0x1002: return "AMD";
        case 0x10DE: return "NVIDIA";
        case 0x8086: return "Intel";
        case 0x106B: return "Apple";
        case 0x13B5: return "ARM";
        case 0x5143: return "Qualcomm";
        default: return "Unknown";
    }
}

bool instance_extension_available(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());
    for (const auto& p : props)
        if (std::strcmp(p.extensionName, name) == 0) return true;
    return false;
}

bool device_extension_available(VkPhysicalDevice phys, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> props(n);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, props.data());
    for (const auto& p : props)
        if (std::strcmp(p.extensionName, name) == 0) return true;
    return false;
}

bool ensure_instance() {
    if (g_instance != VK_NULL_HANDLE) return true;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "lux-accel";
    app.apiVersion = VK_API_VERSION_1_1;

    std::vector<const char*> exts;
    VkInstanceCreateFlags flags = 0;
    // MoltenVK (macOS) requires portability enumeration to surface its device.
    if (instance_extension_available("VK_KHR_portability_enumeration")) {
        exts.push_back("VK_KHR_portability_enumeration");
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.flags = flags;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

    if (vkCreateInstance(&ci, nullptr, &g_instance) != VK_SUCCESS) {
        g_instance = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

void refresh_physical_devices() {
    g_physical_devices.clear();
    if (!g_instance) return;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, nullptr);
    g_physical_devices.resize(n);
    if (n) vkEnumeratePhysicalDevices(g_instance, &n, g_physical_devices.data());
}

bool find_compute_family(VkPhysicalDevice phys, uint32_t* out) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, fams.data());
    for (uint32_t i = 0; i < n; ++i) {
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { *out = i; return true; }
    }
    return false;
}

// Transition the queue's command buffer into the recording state, waiting on
// any prior submission first so the buffer is safe to reset.
void ensure_recording(VulkanQueue* q);

// Referenced by create_buffer_with_data before its definition.
void vulkan_destroy_buffer(void* buffer);

void free_ephemeral(VulkanQueue* q) {
    for (auto b : q->ephemeral_buffers) vkDestroyBuffer(q->dev->device, b, nullptr);
    for (auto m : q->ephemeral_memory) vkFreeMemory(q->dev->device, m, nullptr);
    for (auto p : q->ephemeral_pools) vkDestroyDescriptorPool(q->dev->device, p, nullptr);
    q->ephemeral_buffers.clear();
    q->ephemeral_memory.clear();
    q->ephemeral_pools.clear();
}

void ensure_recording(VulkanQueue* q) {
    if (q->recording) return;
    if (q->submitted) {
        vkWaitForFences(q->dev->device, 1, &q->fence, VK_TRUE, UINT64_MAX);
        vkResetFences(q->dev->device, 1, &q->fence);
        q->submitted = false;
        free_ephemeral(q);
    }
    vkResetCommandBuffer(q->cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(q->cmd, &bi);
    q->recording = true;
}

// ---------------------------------------------------------------------------
// Backend interface implementation
// ---------------------------------------------------------------------------

int vulkan_is_available() {
    if (!ensure_instance()) return 0;
    refresh_physical_devices();
    return g_physical_devices.empty() ? 0 : 1;
}

int vulkan_init() {
    if (!ensure_instance()) return 0;
    refresh_physical_devices();
    return g_physical_devices.empty() ? 0 : 1;
}

void vulkan_shutdown() {
    g_physical_devices.clear();
    if (g_instance) {
        vkDestroyInstance(g_instance, nullptr);
        g_instance = VK_NULL_HANDLE;
    }
}

int vulkan_get_device_count() {
    return static_cast<int>(g_physical_devices.size());
}

int vulkan_get_device_caps(int index, lux_device_caps_t* caps) {
    if (!caps || index < 0 || index >= (int)g_physical_devices.size()) return 0;
    VkPhysicalDevice phys = g_physical_devices[index];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);

    static thread_local std::string name_storage;
    name_storage = props.deviceName;

    caps->name = name_storage.c_str();
    caps->vendor = vendor_name(props.vendorID);
    caps->is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1 : 0;
    caps->is_unified_memory = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 1 : 0;

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint64_t total = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            total += mp.memoryHeaps[i].size;
    caps->total_memory = total;
    caps->max_buffer_size = props.limits.maxStorageBufferRange;
    caps->max_workgroup_size = props.limits.maxComputeWorkGroupInvocations;

    // Subgroup (SIMD) width via the Vulkan 1.1 subgroup properties.
    VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sub;
    vkGetPhysicalDeviceProperties2(phys, &props2);
    caps->simd_width = sub.subgroupSize ? sub.subgroupSize : 32;
    caps->supports_subgroups = sub.subgroupSize ? 1 : 0;

    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceFeatures(phys, &feats);
    caps->supports_fp16 = feats.shaderInt16 ? 1 : 0;
    return 1;
}

void* vulkan_create_device(int index) {
    if (index < 0 || index >= (int)g_physical_devices.size()) return nullptr;
    VkPhysicalDevice phys = g_physical_devices[index];

    uint32_t family = 0;
    if (!find_compute_family(phys, &family)) {
        log_error("no compute-capable queue family");
        return nullptr;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    // MoltenVK exposes a non-conformant subset; the portability_subset device
    // extension must be enabled when the device advertises it.
    std::vector<const char*> dev_exts;
    if (device_extension_available(phys, "VK_KHR_portability_subset"))
        dev_exts.push_back("VK_KHR_portability_subset");

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        log_error("vkCreateDevice failed");
        return nullptr;
    }

    auto* d = new VulkanDevice();
    d->phys = phys;
    d->device = device;
    d->compute_family = family;
    vkGetDeviceQueue(device, family, 0, &d->compute_queue);
    vkGetPhysicalDeviceMemoryProperties(phys, &d->mem_props);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    d->name = props.deviceName;
    return d;
}

void vulkan_destroy_device(void* device) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d) return;
    if (d->device) {
        vkDeviceWaitIdle(d->device);
        vkDestroyDevice(d->device, nullptr);
    }
    delete d;
}

void* vulkan_create_queue(void* device) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d) return nullptr;

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = d->compute_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(d->device, &pci, nullptr, &pool) != VK_SUCCESS) return nullptr;

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(d->device, &ai, &cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(d->device, pool, nullptr);
        return nullptr;
    }

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(d->device, &fci, nullptr, &fence) != VK_SUCCESS) {
        vkDestroyCommandPool(d->device, pool, nullptr);
        return nullptr;
    }

    auto* q = new VulkanQueue();
    q->dev = d;
    q->pool = pool;
    q->cmd = cmd;
    q->fence = fence;
    ensure_recording(q);
    return q;
}

void vulkan_destroy_queue(void* queue) {
    auto* q = static_cast<VulkanQueue*>(queue);
    if (!q) return;
    if (q->submitted) vkWaitForFences(q->dev->device, 1, &q->fence, VK_TRUE, UINT64_MAX);
    free_ephemeral(q);
    if (q->fence) vkDestroyFence(q->dev->device, q->fence, nullptr);
    if (q->pool) vkDestroyCommandPool(q->dev->device, q->pool, nullptr);
    delete q;
}

int vulkan_queue_submit(void* queue) {
    auto* q = static_cast<VulkanQueue*>(queue);
    if (!q) return 0;
    if (q->submitted) return 1;  // already in flight
    if (q->recording) { vkEndCommandBuffer(q->cmd); q->recording = false; }

    vkResetFences(q->dev->device, 1, &q->fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &q->cmd;
    if (vkQueueSubmit(q->dev->compute_queue, 1, &si, q->fence) != VK_SUCCESS) {
        log_error("vkQueueSubmit failed");
        return 0;
    }
    q->submitted = true;
    return 1;
}

int vulkan_queue_wait(void* queue) {
    auto* q = static_cast<VulkanQueue*>(queue);
    if (!q) return 0;
    if (!q->submitted && !vulkan_queue_submit(queue)) return 0;
    vkWaitForFences(q->dev->device, 1, &q->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(q->dev->device, 1, &q->fence);
    q->submitted = false;
    free_ephemeral(q);
    ensure_recording(q);  // ready for the next batch of commands
    return 1;
}

VulkanBuffer* alloc_buffer(VulkanDevice* d, VkDeviceSize size, uint32_t lux_usage) {
    VkBufferUsageFlags vku = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (lux_usage & LUX_BUFFER_USAGE_UNIFORM) vku |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = vku;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(d->device, &bci, nullptr, &buffer) != VK_SUCCESS) return nullptr;

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(d->device, buffer, &req);

    // Host-visible + coherent so map()/unmap() are flush-free; prefer device-local
    // too (true on unified-memory parts such as Apple GPUs via MoltenVK).
    uint32_t mt = find_memory_type(d->mem_props, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) { vkDestroyBuffer(d->device, buffer, nullptr); return nullptr; }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(d->device, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(d->device, buffer, nullptr);
        return nullptr;
    }
    vkBindBufferMemory(d->device, buffer, mem, 0);

    auto* b = new VulkanBuffer();
    b->device = d->device;
    b->buffer = buffer;
    b->memory = mem;
    b->size = size;
    b->usage = lux_usage;
    return b;
}

void* vulkan_create_buffer(void* device, const lux_buffer_desc_t* desc) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d || !desc) return nullptr;
    return alloc_buffer(d, desc->size, desc->usage);
}

void* vulkan_create_buffer_with_data(void* device, const lux_buffer_desc_t* desc, const void* data) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d || !desc || !data) return nullptr;
    auto* b = alloc_buffer(d, desc->size, desc->usage);
    if (!b) return nullptr;
    void* p = nullptr;
    if (vkMapMemory(d->device, b->memory, 0, desc->size, 0, &p) != VK_SUCCESS || !p) {
        log_error("create_buffer_with_data: vkMapMemory failed");
        vulkan_destroy_buffer(b);
        return nullptr;
    }
    std::memcpy(p, data, desc->size);
    vkUnmapMemory(d->device, b->memory);  // coherent memory: no explicit flush
    return b;
}

void vulkan_destroy_buffer(void* buffer) {
    auto* b = static_cast<VulkanBuffer*>(buffer);
    if (!b) return;
    if (b->mapped) vkUnmapMemory(b->device, b->memory);
    if (b->buffer) vkDestroyBuffer(b->device, b->buffer, nullptr);
    if (b->memory) vkFreeMemory(b->device, b->memory, nullptr);
    delete b;
}

void* vulkan_map_buffer(void* buffer) {
    auto* b = static_cast<VulkanBuffer*>(buffer);
    if (!b) return nullptr;
    if (b->mapped) return b->mapped;
    if (vkMapMemory(b->device, b->memory, 0, b->size, 0, &b->mapped) != VK_SUCCESS) {
        log_error("vkMapMemory failed");
        return nullptr;
    }
    return b->mapped;
}

void vulkan_unmap_buffer(void* buffer) {
    auto* b = static_cast<VulkanBuffer*>(buffer);
    if (!b || !b->mapped) return;
    vkUnmapMemory(b->device, b->memory);  // coherent memory: visible without flush
    b->mapped = nullptr;
}

VkShaderModule make_shader_module(VkDevice device, const uint32_t* code, size_t bytes) {
    VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sci.codeSize = bytes;
    sci.pCode = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &sci, nullptr, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    return mod;
}

void* vulkan_create_kernel_from_bundle(void* device, const void* bundle_data,
                                       size_t bundle_size, const char* entry_point) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d || !bundle_data || !entry_point || bundle_size < 4 || (bundle_size % 4) != 0) {
        log_error("create_kernel_from_bundle: need 4-byte-aligned SPIR-V");
        return nullptr;
    }
    VkShaderModule mod = make_shader_module(d->device,
        static_cast<const uint32_t*>(bundle_data), bundle_size);
    if (!mod) { log_error("vkCreateShaderModule failed (SPIR-V)"); return nullptr; }

    auto* k = new VulkanKernel();
    k->device = d->device;
    k->module = mod;
    k->entry = entry_point;
    return k;
}

void* vulkan_create_kernel_from_source(void* device, const char* source, const char* entry_point) {
    auto* d = static_cast<VulkanDevice*>(device);
    if (!d || !source || !entry_point) return nullptr;
#if defined(LUX_HAS_SHADERC) && LUX_HAS_SHADERC
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult res = compiler.CompileGlslToSpv(
        source, std::strlen(source), shaderc_glsl_compute_shader, "kernel.comp", entry_point, options);
    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        log_error(res.GetErrorMessage().c_str());
        return nullptr;
    }
    std::vector<uint32_t> spirv(res.cbegin(), res.cend());
    VkShaderModule mod = make_shader_module(d->device, spirv.data(),
                                            spirv.size() * sizeof(uint32_t));
    if (!mod) { log_error("vkCreateShaderModule failed (compiled GLSL)"); return nullptr; }

    auto* k = new VulkanKernel();
    k->device = d->device;
    k->module = mod;
    k->entry = entry_point;
    return k;
#else
    (void)source; (void)entry_point;
    log_error("create_kernel_from_source: GLSL compilation needs libshaderc "
              "(rebuild lux-vulkan with shaderc, or use create_kernel_from_bundle with SPIR-V)");
    return nullptr;
#endif
}

void destroy_pipeline_cache(VulkanKernel* k) {
    if (k->pipeline) { vkDestroyPipeline(k->device, k->pipeline, nullptr); k->pipeline = VK_NULL_HANDLE; }
    if (k->layout)   { vkDestroyPipelineLayout(k->device, k->layout, nullptr); k->layout = VK_NULL_HANDLE; }
    if (k->dsl)      { vkDestroyDescriptorSetLayout(k->device, k->dsl, nullptr); k->dsl = VK_NULL_HANDLE; }
    k->cached_sig.clear();
}

void vulkan_destroy_kernel(void* kernel) {
    auto* k = static_cast<VulkanKernel*>(kernel);
    if (!k) return;
    destroy_pipeline_cache(k);
    if (k->module) vkDestroyShaderModule(k->device, k->module, nullptr);
    delete k;
}

void vulkan_kernel_set_buffer(void* kernel, uint32_t index, void* buffer, size_t offset) {
    auto* k = static_cast<VulkanKernel*>(kernel);
    auto* b = static_cast<VulkanBuffer*>(buffer);
    if (!k || !b) return;
    VkDeviceSize range = (b->size > offset) ? (b->size - offset) : VK_WHOLE_SIZE;
    k->buffer_args[index] = std::make_tuple(b->buffer, (VkDeviceSize)offset, range);
}

void vulkan_kernel_set_bytes(void* kernel, uint32_t index, const void* data, size_t size) {
    auto* k = static_cast<VulkanKernel*>(kernel);
    if (!k || !data) return;
    k->byte_args[index].assign(static_cast<const uint8_t*>(data),
                               static_cast<const uint8_t*>(data) + size);
}

void vulkan_kernel_set_workgroup_size(void* kernel, uint32_t x, uint32_t y, uint32_t z) {
    auto* k = static_cast<VulkanKernel*>(kernel);
    if (!k) return;
    // Vulkan workgroup (local) size is fixed in the SPIR-V; kept for parity/info.
    k->wg_x = x ? x : 1; k->wg_y = y ? y : 1; k->wg_z = z ? z : 1;
}

// Build (or reuse) the descriptor-set layout + pipeline matching the current
// binding signature: storage buffers for buffer_args, uniform buffers for
// byte_args, all in set 0 at their requested indices.
bool ensure_pipeline(VulkanKernel* k) {
    std::string sig;
    for (const auto& [idx, _] : k->buffer_args) sig += "S" + std::to_string(idx);
    for (const auto& [idx, _] : k->byte_args)   sig += "U" + std::to_string(idx);
    if (k->pipeline != VK_NULL_HANDLE && sig == k->cached_sig) return true;
    destroy_pipeline_cache(k);

    std::vector<VkDescriptorSetLayoutBinding> binds;
    for (const auto& [idx, _] : k->buffer_args) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = idx;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds.push_back(b);
    }
    for (const auto& [idx, _] : k->byte_args) {
        VkDescriptorSetLayoutBinding b{};
        b.binding = idx;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binds.push_back(b);
    }

    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = static_cast<uint32_t>(binds.size());
    dlci.pBindings = binds.empty() ? nullptr : binds.data();
    if (vkCreateDescriptorSetLayout(k->device, &dlci, nullptr, &k->dsl) != VK_SUCCESS) {
        log_error("vkCreateDescriptorSetLayout failed");
        return false;
    }

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &k->dsl;
    if (vkCreatePipelineLayout(k->device, &plci, nullptr, &k->layout) != VK_SUCCESS) {
        log_error("vkCreatePipelineLayout failed");
        return false;
    }

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = k->module;
    stage.pName = k->entry.c_str();

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = stage;
    cpci.layout = k->layout;
    if (vkCreateComputePipelines(k->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &k->pipeline) != VK_SUCCESS) {
        log_error("vkCreateComputePipelines failed");
        return false;
    }
    k->cached_sig = sig;
    return true;
}

int vulkan_dispatch(void* queue, void* kernel, const lux_dispatch_desc_t* desc) {
    auto* q = static_cast<VulkanQueue*>(queue);
    auto* k = static_cast<VulkanKernel*>(kernel);
    if (!q || !k || !desc) return 0;
    if (!ensure_pipeline(k)) return 0;

    VkDevice device = q->dev->device;

    // Descriptor pool sized for this dispatch.
    std::vector<VkDescriptorPoolSize> sizes;
    if (!k->buffer_args.empty())
        sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)k->buffer_args.size()});
    if (!k->byte_args.empty())
        sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, (uint32_t)k->byte_args.size()});

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (!sizes.empty()) {
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.maxSets = 1;
        pci.poolSizeCount = static_cast<uint32_t>(sizes.size());
        pci.pPoolSizes = sizes.data();
        if (vkCreateDescriptorPool(device, &pci, nullptr, &pool) != VK_SUCCESS) {
            log_error("vkCreateDescriptorPool failed");
            return 0;
        }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &k->dsl;
        if (vkAllocateDescriptorSets(device, &dai, &set) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device, pool, nullptr);
            log_error("vkAllocateDescriptorSets failed");
            return 0;
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buf_infos;
    buf_infos.reserve(k->buffer_args.size() + k->byte_args.size());

    for (const auto& [idx, t] : k->buffer_args) {
        buf_infos.push_back({std::get<0>(t), std::get<1>(t), std::get<2>(t)});
    }
    // Ephemeral uniform buffers for byte args (filled now, freed after wait).
    for (const auto& [idx, data] : k->byte_args) {
        VkDeviceSize sz = std::max<VkDeviceSize>(data.size(), 4);
        auto* ub = alloc_buffer(q->dev, sz, LUX_BUFFER_USAGE_UNIFORM);
        if (!ub) { if (pool) vkDestroyDescriptorPool(device, pool, nullptr); return 0; }
        void* m = nullptr;
        if (vkMapMemory(device, ub->memory, 0, sz, 0, &m) == VK_SUCCESS && m) {
            std::memcpy(m, data.data(), data.size());
            if (sz > data.size()) std::memset((uint8_t*)m + data.size(), 0, sz - data.size());
            vkUnmapMemory(device, ub->memory);
        }
        q->ephemeral_buffers.push_back(ub->buffer);
        q->ephemeral_memory.push_back(ub->memory);
        buf_infos.push_back({ub->buffer, 0, sz});
        ub->buffer = VK_NULL_HANDLE;  // ownership transferred to queue cleanup
        ub->memory = VK_NULL_HANDLE;
        delete ub;
    }

    // Emit descriptor writes (buffer_infos vector is stable; index by position).
    size_t pos = 0;
    for (const auto& [idx, t] : k->buffer_args) {
        (void)t;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set; w.dstBinding = idx; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &buf_infos[pos++];
        writes.push_back(w);
    }
    for (const auto& [idx, data] : k->byte_args) {
        (void)data;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set; w.dstBinding = idx; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &buf_infos[pos++];
        writes.push_back(w);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    ensure_recording(q);
    vkCmdBindPipeline(q->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    if (set != VK_NULL_HANDLE)
        vkCmdBindDescriptorSets(q->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout,
                                0, 1, &set, 0, nullptr);
    vkCmdDispatch(q->cmd, desc->grid_x, desc->grid_y, desc->grid_z);

    // Make shader writes visible to subsequent copies, further dispatches, and
    // host reads after the queue drains.
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                       VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(q->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    if (pool) q->ephemeral_pools.push_back(pool);
    return 1;
}

int vulkan_copy_buffer(void* queue, void* src, size_t src_off,
                       void* dst, size_t dst_off, size_t size) {
    auto* q = static_cast<VulkanQueue*>(queue);
    auto* s = static_cast<VulkanBuffer*>(src);
    auto* d = static_cast<VulkanBuffer*>(dst);
    if (!q || !s || !d) return 0;
    ensure_recording(q);
    VkBufferCopy region{(VkDeviceSize)src_off, (VkDeviceSize)dst_off, (VkDeviceSize)size};
    vkCmdCopyBuffer(q->cmd, s->buffer, d->buffer, 1, &region);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(q->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
    return 1;
}

int vulkan_fill_buffer(void* queue, void* buffer, size_t offset, size_t size, uint8_t value) {
    auto* q = static_cast<VulkanQueue*>(queue);
    auto* b = static_cast<VulkanBuffer*>(buffer);
    if (!q || !b) return 0;
    if ((offset % 4) != 0) {
        log_error("fill_buffer: Vulkan requires a 4-byte-aligned offset");
        return 0;
    }
    ensure_recording(q);
    // vkCmdFillBuffer writes 4-byte words; replicate the byte across the word.
    uint32_t word = (uint32_t)value;
    word |= word << 8; word |= word << 16;
    VkDeviceSize fill_size = (size % 4 == 0) ? size : ((size + 3) & ~size_t(3));
    if (offset + fill_size > (size_t)b->size) fill_size = VK_WHOLE_SIZE;
    vkCmdFillBuffer(q->cmd, b->buffer, offset, fill_size, word);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(q->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
    return 1;
}

// ---------------------------------------------------------------------------

static lux_backend_interface_t g_vulkan_interface = {
    .type = LUX_BACKEND_TYPE_VULKAN,
    .name = "vulkan",
    .api_version = LUX_BACKEND_API_VERSION,

    .is_available = vulkan_is_available,
    .init = vulkan_init,
    .shutdown = vulkan_shutdown,

    .get_device_count = vulkan_get_device_count,
    .get_device_caps = vulkan_get_device_caps,

    .create_device = vulkan_create_device,
    .destroy_device = vulkan_destroy_device,

    .create_queue = vulkan_create_queue,
    .destroy_queue = vulkan_destroy_queue,
    .queue_submit = vulkan_queue_submit,
    .queue_wait = vulkan_queue_wait,

    .create_buffer = vulkan_create_buffer,
    .create_buffer_with_data = vulkan_create_buffer_with_data,
    .destroy_buffer = vulkan_destroy_buffer,
    .map_buffer = vulkan_map_buffer,
    .unmap_buffer = vulkan_unmap_buffer,

    .create_kernel_from_source = vulkan_create_kernel_from_source,
    .create_kernel_from_bundle = vulkan_create_kernel_from_bundle,
    .destroy_kernel = vulkan_destroy_kernel,
    .kernel_set_buffer = vulkan_kernel_set_buffer,
    .kernel_set_bytes = vulkan_kernel_set_bytes,
    .kernel_set_workgroup_size = vulkan_kernel_set_workgroup_size,

    .dispatch = vulkan_dispatch,
    .copy_buffer = vulkan_copy_buffer,
    .fill_buffer = vulkan_fill_buffer,
};

#else  // !LUX_HAS_VULKAN — stub path (compiles everywhere; is_available()==0)

int  stub_is_available() { return 0; }
int  stub_init() { return 0; }
void stub_shutdown() {}
int  stub_get_device_count() { return 0; }
int  stub_get_device_caps(int, lux_device_caps_t*) { return 0; }
void* stub_create_device(int) { return nullptr; }
void  stub_destroy_device(void*) {}
void* stub_create_queue(void*) { return nullptr; }
void  stub_destroy_queue(void*) {}
int   stub_queue_submit(void*) { return 0; }
int   stub_queue_wait(void*) { return 0; }
void* stub_create_buffer(void*, const lux_buffer_desc_t*) { return nullptr; }
void* stub_create_buffer_with_data(void*, const lux_buffer_desc_t*, const void*) { return nullptr; }
void  stub_destroy_buffer(void*) {}
void* stub_map_buffer(void*) { return nullptr; }
void  stub_unmap_buffer(void*) {}
void* stub_create_kernel_from_source(void*, const char*, const char*) { return nullptr; }
void* stub_create_kernel_from_bundle(void*, const void*, size_t, const char*) { return nullptr; }
void  stub_destroy_kernel(void*) {}
void  stub_kernel_set_buffer(void*, uint32_t, void*, size_t) {}
void  stub_kernel_set_bytes(void*, uint32_t, const void*, size_t) {}
void  stub_kernel_set_workgroup_size(void*, uint32_t, uint32_t, uint32_t) {}
int   stub_dispatch(void*, void*, const lux_dispatch_desc_t*) { return 0; }
int   stub_copy_buffer(void*, void*, size_t, void*, size_t, size_t) { return 0; }
int   stub_fill_buffer(void*, void*, size_t, size_t, uint8_t) { return 0; }

static lux_backend_interface_t g_vulkan_interface = {
    .type = LUX_BACKEND_TYPE_VULKAN,
    .name = "vulkan",
    .api_version = LUX_BACKEND_API_VERSION,
    .is_available = stub_is_available,
    .init = stub_init,
    .shutdown = stub_shutdown,
    .get_device_count = stub_get_device_count,
    .get_device_caps = stub_get_device_caps,
    .create_device = stub_create_device,
    .destroy_device = stub_destroy_device,
    .create_queue = stub_create_queue,
    .destroy_queue = stub_destroy_queue,
    .queue_submit = stub_queue_submit,
    .queue_wait = stub_queue_wait,
    .create_buffer = stub_create_buffer,
    .create_buffer_with_data = stub_create_buffer_with_data,
    .destroy_buffer = stub_destroy_buffer,
    .map_buffer = stub_map_buffer,
    .unmap_buffer = stub_unmap_buffer,
    .create_kernel_from_source = stub_create_kernel_from_source,
    .create_kernel_from_bundle = stub_create_kernel_from_bundle,
    .destroy_kernel = stub_destroy_kernel,
    .kernel_set_buffer = stub_kernel_set_buffer,
    .kernel_set_bytes = stub_kernel_set_bytes,
    .kernel_set_workgroup_size = stub_kernel_set_workgroup_size,
    .dispatch = stub_dispatch,
    .copy_buffer = stub_copy_buffer,
    .fill_buffer = stub_fill_buffer,
};

#endif // LUX_HAS_VULKAN

} // anonymous namespace

// =============================================================================
// Plugin entry point
// =============================================================================

extern "C" {

LUX_PLUGIN_EXPORT lux_backend_interface_t* lux_backend_init(const lux_core_api_t* core_api) {
    g_core_api = core_api;
    log_info("Vulkan backend plugin loaded");
    return &g_vulkan_interface;
}

} // extern "C"
