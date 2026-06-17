# MIGraphX for Windows — ROCm 7.14 + MIOpen + rocBLAS

MIGraphX 2.16.0.dev built for AMD GPUs (27 architectures, gfx900–gfx1250) on Windows 11.
GPU backend with **MIOpen** (Find-2.0 API), **rocBLAS** (Beta API), **hipBLASLt**, and **hipRTC**.

## Prerequisites

- **Windows 11** (Windows 10 may work, untested)
- **AMD GPU** — any supported arch (see GPU Targets below)
- **Python 3.12** (other 3.10+ versions may work)
- **Visual Studio 2022 BuildTools** — for vcpkg C++ deps + CMake + Ninja
- **Git** — for cloning AMDMIGraphX + rocm-cmake

### Install Python Environment

Create a virtual environment and install the ROCm 7.14 packages.

**AMD Radeon RX 9000 Series (gfx1200/gfx1201):**

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
    "rocm[libraries,device-gfx1201,device-gfx1200]==7.14.0a20260615" `
    "torch[device-gfx1201,device-gfx1200]==2.10.0+rocm7.14.0a20260615" `
    "torchvision[device-gfx1201,device-gfx1200]==0.25.0+rocm7.14.0a20260615" `
    "rocm_sdk_devel==7.14.0a20260615"
```

For other GPU architectures replace `device-gfx1200,device-gfx1201` with your target (e.g. `device-gfx1100` for RX 7000 series).

After install, extract the devel package:
```powershell
$SP = python -c "import site; print(site.getsitepackages()[0])"
New-Item -ItemType Directory -Force -Path rocm-sdk | Out-Null
tar -xf "$SP\rocm_sdk_devel\_devel.tar" -C rocm-sdk
$core  = "$SP\_rocm_sdk_core"
$devel = "rocm-sdk\_rocm_sdk_devel"
Copy-Item "$core\lib\*.lib"      "$devel\lib\"     -Force
Copy-Item "$core\bin\*.dll"      "$devel\bin\"     -Force
Copy-Item "$core\include\*"      "$devel\include\" -Recurse -Force
```

## Quick Start — Python

After building with `build_migraphx.ps1`, `migraphx` is installed into the venv automatically. Use the venv python directly:

```python
import os, torch

# Locate ROCm SDK from torch install (no hardcoded paths)
_sp = os.path.dirname(torch.__file__)
_sp = os.path.dirname(_sp)  # site-packages
if True:
    ROCM_PATH = os.path.join(_sp, "_rocm_sdk_devel")
    os.environ["HIP_PLATFORM"]       = "amd"
    os.environ["HIP_PATH"]           = ROCM_PATH
    os.environ["HIP_CLANG_PATH"]     = os.path.join(ROCM_PATH, "llvm", "bin")
    os.environ["HIP_INCLUDE_PATH"]   = os.path.join(ROCM_PATH, "include")
    os.environ["HIP_LIB_PATH"]       = os.path.join(ROCM_PATH, "lib")
    os.environ["HIP_DEVICE_LIB_PATH"]= os.path.join(ROCM_PATH, "lib", "llvm", "amdgcn", "bitcode")
    os.environ["PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "bin"),
        os.path.join(ROCM_PATH, "llvm", "bin"),
        os.environ.get("PATH", "")
    ])
    os.environ["CPATH"]        = os.path.join(ROCM_PATH, "include") + os.pathsep + os.environ.get("CPATH", "")
    os.environ["LIBRARY_PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "lib"), os.path.join(ROCM_PATH, "lib64"),
        os.environ.get("LIBRARY_PATH", "")
    ])
    os.environ["PKG_CONFIG_PATH"] = os.path.join(ROCM_PATH, "lib", "pkgconfig") + os.pathsep + os.environ.get("PKG_CONFIG_PATH", "")

import migraphx
print(f"MIGraphX {migraphx.__version__}")

# Conv + ReLU on GPU (uses MIOpen for convolution)
p = migraphx.program()
mm = p.get_main_module()
x = mm.add_parameter("x", migraphx.shape(type="float", lens=[1, 3, 224, 224]))
w = mm.add_literal(migraphx.generate_argument(migraphx.shape(type="float", lens=[64, 3, 3, 3])))
c = mm.add_instruction(migraphx.op("convolution", padding=[1, 1], stride=[1, 1]), [x, w])
r = mm.add_instruction(migraphx.op("relu"), [c])
mm.add_return([r])

p.compile(migraphx.get_target("gpu"))
result = p.run({"x": migraphx.generate_argument(migraphx.shape(type="float", lens=[1, 3, 224, 224]))})
print(f"Output: {result[0].get_shape()}")
```

## PyTorch Model Inference

### PyTorch → ONNX → MIGraphX (recommended)

Export from PyTorch, load in MIGraphX for GPU inference.

**Step 1 — Export from PyTorch to ONNX:**

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

**Step 2 — Load & run with MIGraphX:**

```python
import os, numpy as np, torch

_sp = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
ROCM_PATH = os.path.join(_sp, "_rocm_sdk_devel")
os.environ["HIP_PLATFORM"] = "amd"
os.environ["HIP_PATH"]     = ROCM_PATH
os.environ["PATH"] = os.path.join(ROCM_PATH, "llvm", "bin") + os.pathsep + os.environ.get("PATH", "")

import migraphx

model = migraphx.parse_onnx("resnet50.onnx")
model.compile(migraphx.get_target("gpu"))

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
names = list(model.get_parameter_shapes().keys())
result = model.run({names[0]: migraphx.argument(input_data)})
output = np.array(list(result)[0])
print(f"Output: {output.shape}")
```

**Step 3 — Benchmark:**

```python
import time

# Warmup
for _ in range(10):
    model.run({names[0]: migraphx.argument(input_data)})
migraphx.gpu_sync()

N = 100
t0 = time.perf_counter()
for _ in range(N):
    model.run({names[0]: migraphx.argument(input_data)})
migraphx.gpu_sync()
elapsed = time.perf_counter() - t0
print(f"{N/elapsed:.1f} inf/s, {elapsed/N*1000:.2f} ms/inf")
```

### Quantization: FP16 / INT8

```python
model = migraphx.parse_onnx("model.onnx")

# FP16 (fast, small accuracy loss)
migraphx.quantize_fp16(model)
model.compile(migraphx.get_target("gpu"))

# INT8 (fastest, needs calibration)
calib = [migraphx.argument(np.random.randn(1, 3, 224, 224).astype(np.float32))]
migraphx.quantize_int8(model, migraphx.get_target("gpu"), calibration=calib)
model.compile(migraphx.get_target("gpu"))
```

### Save / Load compiled programs

```python
migraphx.save(model, "model.mxr")          # to file
model = migraphx.load("model.mxr")

buf = migraphx.save_buffer(model)          # to bytes
model = migraphx.load_buffer(buf)
```

## Build Configuration

| Feature | Status | Note |
|---------|--------|------|
| GPU backend | ✅ ON | HIP + hipRTC |
| GPU targets | gfx900–gfx1250 (27 arches) | all RDNA/CDNA via multi-arch build |
| **MIOpen** | **✅ ON** | Find-2.0 API + Find Mode API |
| **rocBLAS** | **✅ ON** | Beta API, GEMM acceleration |
| **hipBLASLt** | **✅ ON** | Flexible BLAS |
| Composable Kernel | ❌ OFF | Not ported to Windows |
| MLIR | ❌ OFF | rocMLIR not available on Windows |
| **ONNX parser** | **❌ OFF** | vcpkg protobuf (MSVC) ABI mismatch with clang++; protobuf built with clang++ has cmake target incompatibilities |
| **TF parser** | **❌ OFF** | Same protobuf ABI issue |
| Python bindings | ✅ ON | `migraphx.cp312-win_amd64.pyd` |
| Tests | ❌ OFF | Disabled for faster iteration |

### Enabling ONNX Support

ONNX/TF parsers need protobuf built with clang++ (vcpkg default is MSVC — ABI mismatch). Two paths:

**Path A — Manual protobuf build (recommended):**
```powershell
$ClangCXX = "F:\MIGraphxWin\venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\clang++.exe"
$ClangCC  = "F:\MIGraphxWin\venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\clang.exe"
$Prefix   = "F:\MIGraphxWin\clang-deps"

# Build abseil-cpp
git clone https://github.com/abseil/abseil-cpp.git --depth 1 --branch lts_2024_01_16
cmake -S abseil-cpp -B abseil-cpp/build -G Ninja `
  -DCMAKE_CXX_COMPILER=$ClangCXX -DCMAKE_C_COMPILER=$ClangCC `
  -DCMAKE_INSTALL_PREFIX=$Prefix -DABSL_BUILD_TESTING=OFF
cmake --build abseil-cpp/build --parallel && cmake --install abseil-cpp/build

# Build protobuf
git clone https://github.com/protocolbuffers/protobuf.git --depth 1 --branch v27.0
cmake -S protobuf -B protobuf/build -G Ninja `
  -DCMAKE_CXX_COMPILER=$ClangCXX -DCMAKE_C_COMPILER=$ClangCC `
  -DCMAKE_PREFIX_PATH=$Prefix -DCMAKE_INSTALL_PREFIX=$Prefix `
  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package
cmake --build protobuf/build --parallel && cmake --install protobuf/build
```

Then add `$Prefix` to `CMAKE_PREFIX_PATH` and set `-DMIGRAPHX_ENABLE_ONNX=ON -DMIGRAPHX_ENABLE_TF=ON`.

**Path B — vcpkg custom triplet:**
Create `x64-windows-clang.cmake` triplet with clang compilers, then `vcpkg install protobuf:x64-windows-clang`.

> **⚠️ Current state:** ONNX disabled. `migraphx.parse_onnx()` not available. Use direct API (build programs manually) or rebuild protobuf with clang++ to enable ONNX.

## Building from Source

### Step 1 — Get source + apply patches

```powershell
.\setup_src.ps1
```

Clones AMDMIGraphX at the pinned commit (`0043a53c9`), clones `rocm-cmake`, and applies `patches/windows_build.patch` (Windows compatibility fixes).

### Step 2 — Build

```powershell
# Default: gfx1200 + gfx1201 only (fast)
.\build.bat

# All 27 GPU architectures
.\build_migraphx.ps1 -GPU_TARGETS "gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250"
```

Handles cmake configure, Ninja build, DLL copy, venv install, smoke test, and wheel build.

Test with:
```powershell
F:\MIGraphxWin\venv\Scripts\python.exe test_gpu.py
F:\MIGraphxWin\venv\Scripts\python.exe test_migraphx.py
```

### Patches (`patches/windows_build.patch`)

Applied to AMDMIGraphX source to build on Windows:

| File | Fix |
|------|-----|
| `CMakeLists.txt` | Add `MIGRAPHX_ENABLE_ONNX` / `MIGRAPHX_ENABLE_TF` options |
| `cmake/PythonModules.cmake` | Skip non-numeric Python versions (Astral/uv detection) |
| `src/CMakeLists.txt` | Conditional ONNX/TF subdirs; compile definitions when disabled |
| `src/py/CMakeLists.txt` | Link ONNX/TF libs only when targets exist |
| `src/py/migraphx_py.cpp` | Guard `parse_onnx` / `parse_tf` with `#ifndef MIGRAPHX_DISABLE_*` |
| `src/driver/main.cpp` | Same include guards for ONNX/TF headers |
| `src/targets/gpu/device_name.*` | Fix `#if` guard: `HIPBLASLT \|\| ROCBLAS` |
| `src/targets/gpu/jit/mlir.cpp` | Wrap entire file in `#ifdef MIGRAPHX_MLIR` |

## API Reference

### Program construction
| Function | Description |
|----------|-------------|
| `migraphx.program()` | Create an empty program |
| `p.get_main_module()` | Get main module for adding instructions |
| `p.compile(target)` | Compile for a target (e.g. GPU) |
| `p.run(params)` | Execute with dict of named arguments |
| `p.get_parameter_shapes()` | Get expected input shapes |

### Module instructions
| Method | Description |
|--------|-------------|
| `mm.add_parameter(name, shape)` | Add input placeholder |
| `mm.add_literal(arg)` | Add constant (weights, biases) |
| `mm.add_instruction(op, inputs)` | Add operation node |
| `mm.add_return([...])` | Set module outputs |

### Shapes & arguments
| Function | Description |
|----------|-------------|
| `migraphx.shape(type="float", lens=[1,3,224,224])` | Create shape descriptor |
| `migraphx.generate_argument(shape, seed=0)` | Generate random tensor |
| `migraphx.argument(np_array)` | Wrap numpy array as MIGraphX tensor |
| `np.array(mx_arg)` | Convert MIGraphX tensor to numpy |

### GPU memory
| Function | Description |
|----------|-------------|
| `migraphx.allocate_gpu(shape)` | Allocate GPU buffer |
| `migraphx.to_gpu(arg)` | Copy host→device |
| `migraphx.from_gpu(arg)` | Copy device→host |
| `migraphx.gpu_sync()` | Synchronize GPU stream |

### Common operations
| Op name | Attributes |
|---------|-----------|
| `convolution` | `padding`, `stride`, `dilation`, `group` |
| `pooling` | `mode` (average/max), `padding`, `stride` |
| `relu`, `sigmoid`, `tanh`, `leaky_relu` | — (`leaky_relu`: `alpha`) |
| `softmax` | `axis` |
| `dot` | — (matrix multiply) |
| `add`, `mul`, `sub`, `div` | — (element-wise) |
| `flatten`, `reshape`, `transpose` | `axis` / `dims` / `permutation` |
| `concat` | `axis` |
| `batch_norm_inference` | `epsilon`, `momentum` |

## Files

| File | Purpose |
|------|---------|
| `build_migraphx.ps1` | Full build script (cmake + ninja + DLL copy) |
| `test_migraphx.py` | GPU inference smoke test (pointwise + convolution) |
| `test_gpu.py` | Minimal add+relu GPU test |

## Output Binaries (`build_gpu/bin/`)

### MIGraphX DLLs

| File | Size | Description |
|------|------|-------------|
| `migraphx.dll` | 48 MB | Core library |
| `migraphx_device.dll` | 34 MB | 27-arch HIP kernels (gfx900–gfx1250) |
| `migraphx_gpu.dll` | 4.6 MB | GPU target |
| `migraphx_ref.dll` | 1.1 MB | Reference CPU target |
| `migraphx_py.dll` | 78 KB | Python glue |
| `migraphx_py_3.12.dll` | 219 KB | Python 3.12 bindings |
| `migraphx.cp312-win_amd64.pyd` | 900 KB | Importable Python extension |
| `migraphx-hiprtc-driver.exe` | 108 KB | HipRTC JIT kernel compiler (subprocess) |

### Runtime ROCm DLLs (copied from pip SDK)

| File | Source |
|------|--------|
| `amdhip64_7.dll`, `hiprtc0714.dll`, `hiprtc-builtins0714.dll` | `_rocm_sdk_core/bin` |
| `amd_comgr.dll`, `rocm_kpack.dll` | `_rocm_sdk_core/bin` |
| `MIOpen.dll`, `rocblas.dll`, `libhipblaslt.dll` | `_rocm_sdk_libraries/bin` |
| `sqlite3.dll` | vcpkg |
| `VCRUNTIME140.dll`, `MSVCP140.dll`, `api-ms-win-crt-*.dll` | Windows SDK / System32 |

> The wheel (`dist/migraphx_rocm-*.whl`) bundles only the MIGraphX DLLs. ROCm SDK DLLs are loaded at runtime from the pip packages via a `.pth` setup file — no manual path configuration needed.

## License

MIT — see [AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) upstream.
