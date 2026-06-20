# MIGraphX for Windows — ROCm 7.14 + CK + MLIR + ONNX

MIGraphX 2.x built from source for AMD GPUs (23 architectures, gfx900–gfx1201) on Windows 11.
GPU backend with **Composable Kernel** (CK gemm/attention), **rocMLIR** (conv/GEMM fusion),
**MIOpen**, **rocBLAS**, **hipBLASLt**, and **ONNX Runtime MIGraphX EP**.

**Benchmarked:** 1.92× faster than PyTorch fp32, 2.26× faster fp16 on ResNet-50 (gfx1103, batch 1).

## Prerequisites

- **Windows 11** (Windows 10 may work, untested)
- **AMD GPU** — any supported arch (see GPU Targets below)
- **Python 3.12**
- **Visual Studio 2022 BuildTools** — MSVC compiler + MSBuild (for ONNX Runtime build)
- **Git** — for cloning AMDMIGraphX and dependencies

### Install Python Environment

Create a virtual environment and install ROCm 7.14 packages.

**Example — RDNA3 APU (gfx1103, Ryzen 780M / 760M):**

```powershell
python -m venv .env
.env\Scripts\activate

pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
    "rocm[libraries,device-gfx1103]==7.14.0a20260615" `
    "torch[device-gfx1103]==2.10.0+rocm7.14.0a20260615" `
    "torchvision[device-gfx1103]==0.25.0+rocm7.14.0a20260615" `
    "rocm_sdk_devel==7.14.0a20260615"
```

Replace `device-gfx1103` with your GPU target (e.g. `device-gfx1100` for RX 7900 XTX,
`device-gfx1030` for RX 6800/6900).

## Quick Start — Python

After building with `build-migraphx-win.ps1`, install the produced wheel into your venv:

```powershell
pip install wheels\migraphx-0.1.0+multiarch-cp312-cp312-win_amd64.whl
```

Then use MIGraphX directly:

```python
import migraphx
import numpy as np

# Parse and run an ONNX model on GPU
model = migraphx.parse_onnx("resnet50.onnx")
model.compile(migraphx.get_target("gpu"))

inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
names = list(model.get_parameter_shapes().keys())
result = model.run({names[0]: migraphx.argument(inp)})
output = np.array(list(result)[0])
print(f"Output shape: {output.shape}")
```

> **Note:** `import migraphx` registers the MIGraphX DLL directory. Always import it
> before `import onnxruntime` when using the ONNX Runtime EP (see below).

## PyTorch Model Inference

### PyTorch → ONNX → MIGraphX (recommended)

**Step 1 — Export from PyTorch:**

```python
import torch, torchvision

model = torchvision.models.resnet50(pretrained=True).eval()
dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy, "resnet50.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

**Step 2 — Load and run with MIGraphX:**

```python
import migraphx
import numpy as np

model = migraphx.parse_onnx("resnet50.onnx")
model.compile(migraphx.get_target("gpu"))

inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
names = list(model.get_parameter_shapes().keys())
result = model.run({names[0]: migraphx.argument(inp)})
output = np.array(list(result)[0])
print(output.shape)
```

**Step 3 — Benchmark:**

```python
import time

for _ in range(20):  # warmup
    model.run({names[0]: migraphx.argument(inp)})
migraphx.gpu_sync()

N = 100
t0 = time.perf_counter()
for _ in range(N):
    model.run({names[0]: migraphx.argument(inp)})
migraphx.gpu_sync()
print(f"{N / (time.perf_counter() - t0):.1f} inf/s, "
      f"{(time.perf_counter() - t0) / N * 1000:.2f} ms/inf")
```

### Quantization: FP16 / INT8

```python
model = migraphx.parse_onnx("model.onnx")

# FP16 — fast, minimal accuracy loss
migraphx.quantize_fp16(model)
model.compile(migraphx.get_target("gpu"))

# INT8 — fastest, requires calibration data
calib = [migraphx.argument(np.random.randn(1, 3, 224, 224).astype(np.float32))]
migraphx.quantize_int8(model, migraphx.get_target("gpu"), calibration=calib)
model.compile(migraphx.get_target("gpu"))
```

### Save / Load compiled programs

```python
migraphx.save(model, "model.mxr")      # save to file
model = migraphx.load("model.mxr")     # load from file

buf = migraphx.save_buffer(model)      # save to bytes
model = migraphx.load_buffer(buf)      # load from bytes
```

## ONNX Runtime MIGraphX EP

Install the ORT wheel (delegates subgraphs to MIGraphX, compiled on GPU):

```powershell
pip install wheels\onnxruntime_migraphx-1.28.0-cp312-cp312-win_amd64.whl
```

```python
import migraphx          # MUST come before onnxruntime — registers MIGraphX DLL dirs
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(
    "resnet50.onnx",
    providers=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
)
print("Active providers:", sess.get_providers())

inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = sess.run(None, {"input": inp})
print(out[0].shape)
```

> **Important:** `import migraphx` must precede `import onnxruntime`. MIGraphX's
> `__init__.py` calls `os.add_dll_directory()` for the package dir; without it,
> `migraphx_gpu.dll` fails to load (error 126) when ORT initialises the EP and
> silently falls back to CPU.

## Build Configuration

| Feature | Status | Note |
|---------|--------|------|
| GPU backend | ✓ ON | HIP + hipRTC |
| GPU targets | 23 arches, gfx900–gfx1201 | multi-arch device.dll |
| **Composable Kernel** | **✓ ON** | CK gemm / attention kernels |
| **rocMLIR** | **✓ ON** | Conv + GEMM fusion via MLIR |
| **MIOpen** | **✓ ON** | Convolution, pooling, BN |
| **rocBLAS / hipBLASLt** | **✓ ON** | GEMM acceleration |
| **ONNX parser** | **✓ ON** | Full op coverage via protobuf + abseil |
| **TF parser** | **✓ ON** | TensorFlow frozen graph |
| Python bindings | ✓ ON | `migraphx.cp312-win_amd64.pyd` |
| Tests | ✗ OFF | Disabled for faster build |
| ONNX Runtime EP | ✓ ON | `onnxruntime_providers_migraphx.dll` |

## Building from Source

### Step 1 — Build MIGraphX

```powershell
# Single arch (fast, ~1–2 hr)
.\build-migraphx-win.ps1 -GpuTargets gfx1103

# All 23 supported architectures (~3–5 hr)
.\build-migraphx-win.ps1 -GpuTargets all

# Custom subset
.\build-migraphx-win.ps1 -GpuTargets gfx1100,gfx1103
```

The script handles: cloning all deps, building abseil → protobuf → CK → rocMLIR → MIGraphX,
applying all source patches, and producing the wheel at `wheels/`.

### Step 2 — Build ONNX Runtime MIGraphX EP (optional)

Requires MIGraphX built first (Step 1). From `onnxruntime/` dir (clone `microsoft/onnxruntime`):

```powershell
python tools\ci_build\build.py `
  --build_dir build\win-mgx2 --config Release `
  --use_migraphx --migraphx_home ..\AMDMIGraphX\install `
  --cmake_path .env\Scripts\cmake.exe --ctest_path .env\Scripts\ctest.exe `
  --build_shared_lib --enable_pybind --build_wheel `
  --cmake_extra_defines `
    CMAKE_PREFIX_PATH=.env\Lib\site-packages\_rocm_sdk_devel `
    onnxruntime_BUILD_UNIT_TESTS=OFF `
    FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER `
  --skip_submodule_sync --parallel --update --build
```

**Three patches applied to `cmake/onnxruntime_providers_migraphx.cmake`:**

1. **hiprtc/comgr DLL versions** — add `amd_comgr.dll`, `hiprtc0714.dll`,
   `hiprtc-builtins0714.dll` to the Windows DLL copy list (upstream hardcodes 0602/0604/0700).
2. **`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER`** — prevents the ROCm SDK's
   `nlohmann_json` (lacks `.natvis`) from shadowing ORT's own fetched copy.
3. **`/std:c++17` for the EP target** — `migraphx.hpp` uses `module` as a C++ type name;
   MSVC C++20 mode treats it as a keyword (C2059/C7586). Setting `/std:c++17` on
   `onnxruntime_providers_migraphx` only fixes this without affecting the rest of ORT.

### GPU Targets

```
# RDNA4
gfx1201 (RX 9070 XT)   gfx1200 (RX 9070)
# RDNA3.5
gfx1153  gfx1152  gfx1151  gfx1150
# RDNA3
gfx1103 (780M/760M APU)  gfx1102  gfx1101 (RX 7800 XT)  gfx1100 (RX 7900 XTX)
# RDNA2
gfx1036  gfx1035  gfx1034  gfx1033  gfx1032  gfx1031  gfx1030 (RX 6800/6900)
# RDNA1
gfx1012  gfx1011  gfx1010
# Vega / GCN5
gfx906 (Vega 20)  gfx90c  gfx900 (Vega 10)
```

### Source Patches the Script Auto-Applies

| # | File | Fix |
|---|------|-----|
| 1 | `composable_kernel/CMakeLists.txt` | `/std:c++20` not `-std=c++20` (clang-cl); placed before add_embed_library |
| 2 | `composable_kernel/include/ck/utility/utils.hpp` | Add missing `#include <string>` |
| 3 | CK codegen cmake | `-DEMBED_USE=CArrays` (RC mode fails on long Windows paths) |
| 4 | rocMLIR cmake | `-DLLVM_DISABLE_ASSEMBLY_FILES=ON` (ml64.exe chokes on clang flags in BLAKE3 .asm) |
| 5 | `AMDMIGraphX/src/targets/gpu/ck.hpp` | Remove `#ifndef _WIN32` guard around `MIGRAPHX_DECLARE_ENV_VAR` for CK env vars |
| 6 | git config | `core.longpaths=true` (CK repo has very long paths) |

### Non-Obvious Build Gotchas

| Issue | Fix |
|-------|-----|
| abseil ABI mismatch (`absl::string_view` vs `std::string_view`) | Pin `ABSL_OPTION_USE_STD_STRING_VIEW=1` in `absl/base/options.h` before building abseil; rebuild abseil + protobuf together |
| HIP multi-arch offload-bundle bug (arches silently dropped in device.dll) | `-DMIGRAPHX_WORKAROUND_HIP_MULTI_ARCH_BUG=ON` |
| protoc `IMPORTED_LOCATION` blank → `protoc --cpp_out :path` emitted literally | Append `include(protoc-prebuilt-fix.cmake)` to `protobuf-config.cmake` |
| GNU tar reads `C:\…` as remote host | Pass `--force-local` to tar |
| git clone stderr → PowerShell `NativeCommandError` under `ErrorActionPreference=Stop` | Wrap all git calls in a helper that temporarily sets `ErrorActionPreference = Continue` |
| Stale `CMakeCache.txt` from a prior single-arch build poisons reconfigure | Wipe `CMakeCache.txt + CMakeFiles/` when `.configure_done` marker is absent |

## API Reference

### Program construction

| Function | Description |
|----------|-------------|
| `migraphx.program()` | Create an empty program |
| `p.get_main_module()` | Get main module for adding instructions |
| `p.compile(target)` | Compile for a target (`migraphx.get_target("gpu")`) |
| `p.run(params)` | Execute with dict of named arguments |
| `p.get_parameter_shapes()` | Get expected input shapes |

### Module instructions

| Method | Description |
|--------|-------------|
| `mm.add_parameter(name, shape)` | Add named input placeholder |
| `mm.add_literal(np_array)` | Add constant tensor (weights, biases) |
| `mm.add_instruction(op, inputs)` | Add operation node |
| `mm.add_return([...])` | Set module outputs |

### Shapes and arguments

| Function | Description |
|----------|-------------|
| `migraphx.shape(type="float_type", lens=[1,3,224,224])` | Create shape descriptor |
| `migraphx.argument(np_array)` | Wrap numpy array as MIGraphX tensor |
| `np.array(mx_result)` | Convert MIGraphX result tensor to numpy |
| `migraphx.generate_argument(shape)` | Generate random tensor for testing |

### Quantization

| Function | Description |
|----------|-------------|
| `migraphx.quantize_fp16(program)` | Convert ops to FP16 in-place |
| `migraphx.quantize_int8(program, target, calibration=[...])` | INT8 with calibration data |

### GPU utilities

| Function | Description |
|----------|-------------|
| `migraphx.gpu_sync()` | Synchronize GPU stream (use before timing) |
| `migraphx.get_target("gpu")` | Get GPU compile target |
| `migraphx.get_target("ref")` | Get CPU reference target |

### Common operations

| Op name | Key attributes |
|---------|----------------|
| `convolution` | `padding`, `stride`, `dilation`, `group` |
| `pooling` | `mode` (average/max), `padding`, `stride` |
| `relu`, `sigmoid`, `tanh`, `leaky_relu` | — (`leaky_relu`: `alpha`) |
| `softmax` | `axis` |
| `dot` | — (matrix multiply) |
| `add`, `mul`, `sub`, `div` | — (element-wise) |
| `reshape`, `transpose`, `flatten` | `dims` / `permutation` / `axis` |
| `concat` | `axis` |
| `batch_norm_inference` | `epsilon`, `momentum` |

## Files

| File | Purpose |
|------|---------|
| `build-migraphx-win.ps1` | Full build script (PowerShell) |
| `build-migraphx-win.sh` | Full build script (Git Bash) |
| `make-wheel.ps1` | Package built artifacts into a pip wheel |
| `benchmark_migraphx.py` | ResNet-50 benchmark vs PyTorch |
| `wheels/` | Built wheels (migraphx + onnxruntime_migraphx) |

## Output Binaries (`AMDMIGraphX/build/bin/`)

### MIGraphX DLLs

| File | Size | Description |
|------|------|-------------|
| `migraphx.dll` | 46 MB | Core library |
| `migraphx_device.dll` | 30 MB | 23-arch HIP kernels (gfx900–gfx1201) |
| `migraphx_gpu.dll` | 115 MB | GPU target (CK + MLIR + MIOpen + rocBLAS) |
| `migraphx_onnx.dll` | 3.1 MB | ONNX graph parser |
| `migraphx_tf.dll` | 2.6 MB | TensorFlow graph parser |
| `migraphx_ref.dll` | 1.0 MB | Reference CPU target |
| `migraphx_c.dll` | 370 KB | C API (used by ONNX Runtime EP) |
| `migraphx_py.dll` | 70 KB | Python glue |
| `migraphx_py_3.12.dll` | 165 KB | Python 3.12 bindings |
| `migraphx.cp312-win_amd64.pyd` | 730 KB | Importable Python extension |
| `migraphx-hiprtc-driver.exe` | 94 KB | HipRTC JIT kernel compiler |

### ONNX Runtime DLLs (`onnxruntime/build/win-mgx2/Release/Release/`)

| File | Size | Description |
|------|------|-------------|
| `onnxruntime.dll` | 16 MB | ORT core |
| `onnxruntime_providers_migraphx.dll` | 357 KB | MIGraphX execution provider |
| `onnxruntime_providers_shared.dll` | 11 KB | Shared EP loader |

### Wheels (`wheels/`)

| File | Size | Description |
|------|------|-------------|
| `migraphx-0.1.0+multiarch-cp312-cp312-win_amd64.whl` | 48 MB | MIGraphX — all 23 GPU arches |
| `onnxruntime_migraphx-1.28.0-cp312-cp312-win_amd64.whl` | 60 MB | ORT with MIGraphX EP |

> Wheels bundle MIGraphX DLLs only. ROCm runtime DLLs (`amdhip64.dll`, `MIOpen.dll`,
> `rocblas.dll`, etc.) are loaded at runtime from the `rocm_sdk_*` pip packages —
> no manual PATH configuration needed.

## Benchmark

ResNet-50, batch 1, 224×224, 100 iterations after 20 warmup — gfx1103 (Ryzen AI 780M):

| Backend | Precision | Mean (ms) | vs PyTorch |
|---------|-----------|-----------|------------|
| MIGraphX GPU | fp32 | 47.9 | **1.92×** |
| MIGraphX GPU | fp16 | 31.9 | **2.26×** |
| PyTorch GPU | fp32 | 91.9 | 1.0× |
| PyTorch GPU | fp16 | 72.0 | 1.0× |

## License

MIT — see [AMDMIGraphX upstream](https://github.com/ROCm/AMDMIGraphX).
