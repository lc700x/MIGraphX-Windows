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
print(f"GPU target: {migraphx.get_target('gpu')}")

# Test 1: pointwise (add + relu)
print("\n--- Test 1: Pointwise ops ---")
p = migraphx.program()
mm = p.get_main_module()
s = migraphx.shape(type='float', lens=[1, 64, 32, 32])
x = mm.add_parameter('x', s)
y = mm.add_parameter('y', s)
z = mm.add_instruction(migraphx.op('add'), [x, y])
r = mm.add_instruction(migraphx.op('relu'), [z])
mm.add_return([r])
p.compile(migraphx.get_target('gpu'))
result = p.run({'x': migraphx.generate_argument(s), 'y': migraphx.generate_argument(s)})
print(f"  Output: {result[0].get_shape()}")

# Test 2: convolution (uses MIOpen + rocBLAS)
print("\n--- Test 2: Convolution (MIOpen + rocBLAS) ---")
p2 = migraphx.program()
mm2 = p2.get_main_module()
s2 = migraphx.shape(type='float', lens=[1, 3, 224, 224])
x2 = mm2.add_parameter('x', s2)
w2 = mm2.add_literal(migraphx.generate_argument(migraphx.shape(type='float', lens=[64, 3, 3, 3])))
c2 = mm2.add_instruction(migraphx.op('convolution', padding=[1, 1], stride=[1, 1]), [x2, w2])
r2 = mm2.add_instruction(migraphx.op('relu'), [c2])
mm2.add_return([r2])
p2.compile(migraphx.get_target('gpu'))
result2 = p2.run({'x': migraphx.generate_argument(s2)})
print(f"  Output: {result2[0].get_shape()}")

print("\n=== MIGraphX GPU with MIOpen + rocBLAS + hipBLASLt — ALL WORKING ===")
