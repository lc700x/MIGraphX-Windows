import os

# os.add_dll_directory(r"F:\MIGraphxWin\venv\Lib\site-packages\_rocm_sdk_libraries\bin")
# os.add_dll_directory(r"F:\MIGraphxWin\venv\Lib\site-packages\_rocm_sdk_core\bin")
if True:
    import torch
    # Set up ROCm7 Path
    torch_dir = os.path.dirname(torch.__file__)
    site_packages = os.path.dirname(torch_dir)
    ROCM_PATH = os.path.join(site_packages, "_rocm_sdk_devel")

    os.environ["HIP_PLATFORM"] = "amd"
    os.environ["HIP_PATH"] = ROCM_PATH
    os.environ["HIP_CLANG_PATH"] = os.path.join(ROCM_PATH, "llvm", "bin")
    os.environ["HIP_INCLUDE_PATH"] = os.path.join(ROCM_PATH, "include")
    os.environ["HIP_LIB_PATH"] = os.path.join(ROCM_PATH, "lib")
    os.environ["HIP_DEVICE_LIB_PATH"] = os.path.join(ROCM_PATH, "lib", "llvm", "amdgcn", "bitcode")
    os.environ["PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "bin"),
        os.path.join(ROCM_PATH, "llvm", "bin"),
        os.environ.get("PATH", "")
    ])
    os.environ["CPATH"] = os.path.join(ROCM_PATH, "include") + os.pathsep + os.environ.get("CPATH", "")
    os.environ["LIBRARY_PATH"] = os.pathsep.join([
        os.path.join(ROCM_PATH, "lib"),
        os.path.join(ROCM_PATH, "lib64"),
        os.environ.get("LIBRARY_PATH", "")
    ])
    os.environ["PKG_CONFIG_PATH"] = os.path.join(ROCM_PATH, "lib", "pkgconfig") + os.pathsep + os.environ.get("PKG_CONFIG_PATH", "")
    
import migraphx
print(f"MIGraphX {migraphx.__version__}")

# Test: add + relu
p = migraphx.program()
mm = p.get_main_module()
s = migraphx.shape(type='float', lens=[1, 64, 32, 32])
x = mm.add_parameter('x', s)
y = mm.add_parameter('y', s)
z = mm.add_instruction(migraphx.op('add'), [x, y])
r = mm.add_instruction(migraphx.op('relu'), [z])
mm.add_return([r])

print("Compiling...")
p.compile(migraphx.get_target('gpu'))
print("Compiled! Running...")
result = p.run({'x': migraphx.generate_argument(s), 'y': migraphx.generate_argument(s)})
print(f"Inference OK: output shape {result[0].get_shape()}")
