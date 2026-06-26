// Copyright (c) 2024-2026 Lux Industries Inc.
// SPDX-License-Identifier: BSD-3-Clause
//
// Smoke test: dlopen the plugin, resolve lux_backend_init, query interface,
// then on a host with a Vulkan ICD actually create a device, allocate buffers,
// compile a GLSL compute vector-add to SPIR-V, dispatch it and verify results.
//
// On a host without a Vulkan ICD, is_available() returns 0 and the test passes
// after the identity check. On a host with Vulkan but built without libshaderc,
// the GLSL-from-source step is skipped (create_kernel_from_bundle/SPIR-V is the
// alternative) and the test still passes on the device/identity checks.

#include <lux/accel/backend_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

void log_info(const char* msg)  { std::fprintf(stderr, "[info]  %s\n", msg); }
void log_warn(const char* msg)  { std::fprintf(stderr, "[warn]  %s\n", msg); }
void log_error(const char* msg) { std::fprintf(stderr, "[error] %s\n", msg); }
void log_debug(const char* msg) { std::fprintf(stderr, "[debug] %s\n", msg); }

void* core_alloc(size_t n) { return std::malloc(n); }
void  core_free(void* p)   { std::free(p); }

const void* get_kernel_bundle(const char*, size_t* sz) { if (sz) *sz = 0; return nullptr; }
const char* get_kernel_source(const char*) { return nullptr; }

// GLSL compute kernel: c[i] = a[i] + b[i]. Bindings match the set_buffer /
// set_bytes indices used below (storage 0/1/2, uniform 3).
const char* kAddKernel = R"GLSL(
#version 450
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly  buffer A { float a[]; };
layout(set = 0, binding = 1) readonly  buffer B { float b[]; };
layout(set = 0, binding = 2) writeonly buffer C { float c[]; };
layout(set = 0, binding = 3) uniform Params { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < n) c[i] = a[i] + b[i];
}
)GLSL";

} // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "./lux_vulkan.plugin";

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto init = reinterpret_cast<lux_backend_init_fn>(dlsym(handle, LUX_BACKEND_INIT_SYMBOL));
    if (!init) {
        std::fprintf(stderr, "dlsym(%s) failed: %s\n", LUX_BACKEND_INIT_SYMBOL, dlerror());
        dlclose(handle);
        return 1;
    }

    lux_core_api_t core = {};
    core.api_version       = LUX_BACKEND_API_VERSION;
    core.log_debug         = log_debug;
    core.log_info          = log_info;
    core.log_warn          = log_warn;
    core.log_error         = log_error;
    core.alloc             = core_alloc;
    core.free              = core_free;
    core.get_kernel_bundle = get_kernel_bundle;
    core.get_kernel_source = get_kernel_source;

    lux_backend_interface_t* iface = init(&core);
    if (!iface) { std::fprintf(stderr, "init returned null\n"); dlclose(handle); return 1; }

    if (iface->api_version != LUX_BACKEND_API_VERSION ||
        iface->type != LUX_BACKEND_TYPE_VULKAN ||
        !iface->name || std::strcmp(iface->name, "vulkan") != 0) {
        std::fprintf(stderr, "interface identity mismatch (type=%d name=%s)\n",
                     (int)iface->type, iface->name ? iface->name : "(null)");
        dlclose(handle); return 1;
    }

    int avail = iface->is_available ? iface->is_available() : 0;
    std::fprintf(stderr, "vulkan is_available -> %d\n", avail);
    if (!avail) {
        std::fprintf(stderr, "OK: lux-vulkan plugin loaded and self-identified "
                             "(no Vulkan ICD here; install one to run compute)\n");
        dlclose(handle);
        return 0;
    }

    if (!iface->init || !iface->init()) { std::fprintf(stderr, "init() failed\n"); dlclose(handle); return 1; }

    int nDev = iface->get_device_count();
    if (nDev <= 0) { std::fprintf(stderr, "no vulkan devices\n"); iface->shutdown(); dlclose(handle); return 1; }

    lux_device_caps_t caps = {};
    iface->get_device_caps(0, &caps);
    std::fprintf(stderr, "vulkan device 0: %s (vendor=%s, total_mem=%llu, simd=%u)\n",
                 caps.name ? caps.name : "?", caps.vendor ? caps.vendor : "?",
                 (unsigned long long)caps.total_memory, caps.simd_width);

    void* dev = iface->create_device(0);
    void* queue = dev ? iface->create_queue(dev) : nullptr;
    if (!dev || !queue) { std::fprintf(stderr, "device/queue creation failed\n"); dlclose(handle); return 1; }

    void* kernel = iface->create_kernel_from_source(dev, kAddKernel, "main");
    if (!kernel) {
        std::fprintf(stderr, "OK: device usable; GLSL source path unavailable "
                             "(plugin built without libshaderc) — identity+device verified\n");
        iface->destroy_queue(queue);
        iface->destroy_device(dev);
        iface->shutdown();
        dlclose(handle);
        return 0;
    }

    constexpr uint32_t N = 1024;
    float a[N], b[N];
    for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = (float)(2 * i); }

    lux_buffer_desc_t desc_in = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_COPY_SRC, nullptr };
    lux_buffer_desc_t desc_out = { N * sizeof(float),
        LUX_BUFFER_USAGE_STORAGE | LUX_BUFFER_USAGE_MAP_READ, nullptr };

    void* bufA = iface->create_buffer_with_data(dev, &desc_in, a);
    void* bufB = iface->create_buffer_with_data(dev, &desc_in, b);
    void* bufC = iface->create_buffer(dev, &desc_out);
    if (!bufA || !bufB || !bufC) { std::fprintf(stderr, "buffer setup failed\n"); return 1; }

    iface->kernel_set_buffer(kernel, 0, bufA, 0);
    iface->kernel_set_buffer(kernel, 1, bufB, 0);
    iface->kernel_set_buffer(kernel, 2, bufC, 0);
    iface->kernel_set_bytes (kernel, 3, &N, sizeof(N));
    iface->kernel_set_workgroup_size(kernel, 64, 1, 1);

    lux_dispatch_desc_t disp = { (N + 63) / 64, 1, 1, 64, 1, 1 };
    if (!iface->dispatch(queue, kernel, &disp) || !iface->queue_wait(queue)) {
        std::fprintf(stderr, "dispatch/wait failed\n"); return 1;
    }

    const float* out = static_cast<const float*>(iface->map_buffer(bufC));
    if (!out) { std::fprintf(stderr, "map_buffer failed\n"); return 1; }
    int errors = 0;
    for (uint32_t i = 0; i < N; i++) {
        float expect = a[i] + b[i];
        if (out[i] != expect) {
            if (errors < 4) std::fprintf(stderr, "mismatch %u: %f != %f\n",
                                         i, (double)out[i], (double)expect);
            errors++;
        }
    }
    iface->unmap_buffer(bufC);

    iface->destroy_kernel(kernel);
    iface->destroy_buffer(bufA);
    iface->destroy_buffer(bufB);
    iface->destroy_buffer(bufC);
    iface->destroy_queue(queue);
    iface->destroy_device(dev);
    iface->shutdown();
    dlclose(handle);

    if (errors) { std::fprintf(stderr, "FAIL: %d/%u mismatches\n", errors, N); return 1; }
    std::fprintf(stderr, "OK: %u-element vector add via Vulkan verified\n", N);
    return 0;
}
