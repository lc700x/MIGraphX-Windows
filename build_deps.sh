#!/bin/bash
# Build abseil-cpp + protobuf with clang++ for ONNX/TF parser support.
#
# Key constraints discovered:
#  - Must use /MD (dynamic CRT) to match MIGraphX's clang++ build, set via
#    CMAKE_MSVC_RUNTIME_LIBRARY toolchain file (protobuf's own option forces /MT).
#  - protobuf-targets.cmake wraps utf8_range in $<LINK_ONLY:...> which does NOT
#    propagate to dependents through a STATIC IMPORTED target on Windows/lld —
#    strip the wrapper post-install so utf8_validity.lib actually links.
#  - clang-deps must come FIRST in MIGraphX's CMAKE_PREFIX_PATH so this clang
#    abseil wins over vcpkg's MSVC abseil (same target names, different ABI).
set -e

CMAKE_EXE="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
RC_EXE="C:/Program Files/AMD/ROCm/7.1/bin/llvm-rc.exe"
NINJA_EXE="F:/MIGraphxWin/vcpkg/downloads/tools/ninja-1.13.2-windows/ninja.exe"
PREFIX="F:/MIGraphxWin/clang-deps"

export CC="/F/MIGraphxWin/venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/amdclang.exe"
export CXX="/F/MIGraphxWin/venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/amdclang++.exe"
export PATH="/F/MIGraphxWin/vcpkg/downloads/tools/ninja-1.13.2-windows:$PATH"

# /MD (dynamic CRT) toolchain — matches MIGraphX clang++ build
cat > /tmp/md-toolchain.cmake << 'EOF'
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
EOF

echo "=== Abseil (clang++, /MD) ==="
rm -rf F:/MIGraphxWin/abseil-cpp/build
"$CMAKE_EXE" -S F:/MIGraphxWin/abseil-cpp -B F:/MIGraphxWin/abseil-cpp/build -G Ninja \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DCMAKE_RC_COMPILER="$RC_EXE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/tmp/md-toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DABSL_BUILD_TESTING=OFF \
  -DCMAKE_MAKE_PROGRAM="$NINJA_EXE"
"$CMAKE_EXE" --build F:/MIGraphxWin/abseil-cpp/build --parallel 8
"$CMAKE_EXE" --install F:/MIGraphxWin/abseil-cpp/build

echo "=== Protobuf (clang++, /MD) ==="
rm -rf F:/MIGraphxWin/protobuf/build
"$CMAKE_EXE" -S F:/MIGraphxWin/protobuf -B F:/MIGraphxWin/protobuf/build -G Ninja \
  -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" -DCMAKE_RC_COMPILER="$RC_EXE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_TOOLCHAIN_FILE=/tmp/md-toolchain.cmake \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_ABSL_PROVIDER=package -Dprotobuf_BUILD_PROTOC_BINARIES=ON \
  -Dprotobuf_MSVC_STATIC_RUNTIME=OFF \
  -DCMAKE_MAKE_PROGRAM="$NINJA_EXE"
"$CMAKE_EXE" --build F:/MIGraphxWin/protobuf/build --parallel 8
"$CMAKE_EXE" --install F:/MIGraphxWin/protobuf/build

echo "=== Patch protobuf-targets.cmake (utf8_range LINK_ONLY -> plain) ==="
# $<LINK_ONLY:utf8_range::utf8_validity> does not propagate through STATIC
# IMPORTED protobuf on lld-link; drop the wrapper so utf8_validity.lib links.
sed -i 's|\\\$<LINK_ONLY:utf8_range::utf8_validity>|utf8_range::utf8_validity|g' \
  "$PREFIX/lib/cmake/protobuf/protobuf-targets.cmake"

echo "=== Done — clang-deps ready at $PREFIX ==="
