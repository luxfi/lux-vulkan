# lux-vulkan — Vulkan (SPIR-V) compute backend plugin for lux-accel

`lux-vulkan` (CMake project `lux-vulkan`, v0.1.x) is a **plugin** (shared library
/ CMake `MODULE`, output `lux_vulkan.plugin`) that `lux-accel` loads at runtime
via `dlopen` + `lux_backend_init`. It links nothing from lux-accel — only the
`backend_api.h` header. It targets core **Vulkan 1.1 compute**.

On systems without a Vulkan loader/headers it builds as a no-op stub
(`is_available()==0`).

## Where things are

- `src/vulkan_plugin.cpp` — the whole backend (instance/device/queue/buffer/
  kernel/dispatch over Vulkan compute).
  - Kernels are SPIR-V. `create_kernel_from_bundle` consumes a SPIR-V binary
    directly (`vkCreateShaderModule`); `create_kernel_from_source` compiles GLSL
    compute → SPIR-V at runtime via **libshaderc** when `LUX_HAS_SHADERC` is set
    (the analogue of NVRTC for CUDA / MSL for Metal).
  - Storage buffers bind as `STORAGE_BUFFER`, `set_bytes` args as ephemeral
    `UNIFORM_BUFFER`s, all in set 0 at the requested indices. The pipeline +
    descriptor-set layout are built lazily from the binding signature and cached.
  - Buffers are `HOST_VISIBLE | HOST_COHERENT` (preferring `DEVICE_LOCAL`), so
    `map/unmap` are flush-free; a compute→host/transfer memory barrier after
    each dispatch makes results visible to readback and chained ops.
  - The queue keeps one recording command buffer + fence; `submit` flushes,
    `wait` drains and frees per-submission ephemeral resources.
- `test/test_load.cpp` — load/identity test; runs a real GLSL vector-add
  dispatch when a Vulkan ICD is present.

## Runs live on macOS via MoltenVK

Unlike HIP, Vulkan **runs on this Mac** through MoltenVK (Vulkan → Metal). The
plugin builds against Homebrew `vulkan-headers` + `vulkan-loader` and `shaderc`,
and executes real compute on Apple GPUs. Verified: `Apple M1 Max`, 1024- and
4096-element vector-add, bit-identical to the CPU reference and to the Metal
backend.

```bash
brew install vulkan-headers vulkan-loader molten-vk shaderc
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLUXACCEL_INCLUDE_DIR=/path/to/lux-accel/include
cmake --build build -j

# MoltenVK is an ICD the loader must be pointed at:
export VK_ICD_FILENAMES=/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/molten-vk/lib:/opt/homebrew/opt/vulkan-loader/lib
ctest --test-dir build --output-on-failure   # runs the Vulkan vector-add
```

On Linux with a native ICD (AMD/NVIDIA/Intel) it runs unchanged; the
`VK_KHR_portability_*` extensions are negotiated only when present (MoltenVK).

## Registration

Registered in lux-accel's enums as `LUX_BACKEND_TYPE_VULKAN=5`
(`backend_api.h`), `LUX_BACKEND_VULKAN=5` (`c_api.h`), and
`lux::gpu::BackendType::Vulkan` (`lux-gpu/gpu.hpp`). The cross-backend
conformance harness (`lux-gpu-conformance`) lists it as one row in `kBackends[]`.
