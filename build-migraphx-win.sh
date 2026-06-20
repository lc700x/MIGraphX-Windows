#!/bin/bash
# ============================================================================
# Build MIGraphX from source on Windows (Git Bash / MSYS2)
# GPU + CK (Composable Kernel) + MLIR + ONNX + Python bindings
#
# Prerequisites:
#   - Windows 10/11 x64
#   - Python 3.12 with venv
#   - Git for Windows (with Git Bash)
#   - Visual Studio 2022 Build Tools (for Windows SDK rc.exe)
#   - Internet connection (clones repos, downloads deps)
#
# Usage:
#   ./build-migraphx-win.sh [GPU_TARGETS...]
#
# Examples:
#   ./build-migraphx-win.sh gfx1103                    # RDNA3 (RX 7600, Phoenix APU)
#   ./build-migraphx-win.sh gfx1100                    # RDNA3 (RX 7900 XTX)
#   ./build-migraphx-win.sh gfx1030                    # RDNA2 (RX 6800)
#   ./build-migraphx-win.sh gfx942                     # MI300
#   ./build-migraphx-win.sh gfx1100 gfx1103            # multi-GPU (space separated)
#   ./build-migraphx-win.sh gfx1100 gfx1103 gfx942     # 3 targets
#   ./build-migraphx-win.sh all                         # all common consumer + DC arches
#
# Common GPU architectures:
#   RDNA3:   gfx1100 (7900XTX) gfx1101 (7800XT) gfx1102 (7700XT) gfx1103 (780M/760M APU)
#   RDNA2:   gfx1030 (6800/6900) gfx1031 (6600) gfx1032 (6500)
#   CDNA3:   gfx942 (MI300X/MI300A)
#   CDNA2:   gfx90a (MI250X/MI210)
#   CDNA1:   gfx908 (MI100)
#
# WARNING: Each GPU target adds ~20-40 min of HIP kernel compilation time.
#
# The script is idempotent — safe to re-run after fixing errors.
# ============================================================================
set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Parse GPU targets: support space-separated args, "all" keyword, or default
ALL_TARGETS="gfx1100;gfx1101;gfx1102;gfx1103;gfx1030;gfx1031;gfx942;gfx90a;gfx908"
if [ "$1" = "all" ]; then
    GPU_TARGETS="$ALL_TARGETS"
elif [ $# -gt 1 ]; then
    # Multiple args -> join with semicolons
    GPU_TARGETS=$(IFS=';'; echo "$*")
elif [ $# -eq 1 ]; then
    GPU_TARGETS="$1"
else
    GPU_TARGETS="gfx1103"
fi
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.env"
MIGRAPHX_SRC="$ROOT_DIR/AMDMIGraphX"
CK_SRC="$ROOT_DIR/composable_kernel"
ROCMLIR_SRC="$ROOT_DIR/rocMLIR"
CGET_PREFIX="$MIGRAPHX_SRC/cget"
INSTALL_DIR="$MIGRAPHX_SRC/install"
JOBS="${BUILD_JOBS:-$(nproc)}"

# Pinned commits (from MIGraphX requirements.txt)
CK_COMMIT="ad0db05b040bacda751c65c705261b8a0a7ed25d"
ROCMLIR_COMMIT="364015202c7271708f6375f34eaf20c2a9c199a3"

# Windows SDK path for rc.exe — adjust if your SDK version differs
WIN_SDK_BIN="/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"

echo "============================================"
echo "MIGraphX Windows Build Script"
echo "GPU targets: $GPU_TARGETS"
echo "Root dir:    $ROOT_DIR"
echo "Jobs:        $JOBS"
echo "============================================"

# ---------------------------------------------------------------------------
# Step 0: Validate Windows SDK (rc.exe needed for protobuf)
# ---------------------------------------------------------------------------
if [ ! -f "$WIN_SDK_BIN/rc.exe" ]; then
    echo "ERROR: rc.exe not found at $WIN_SDK_BIN/rc.exe"
    echo "Install Visual Studio Build Tools with Windows SDK component."
    echo "Or set WIN_SDK_BIN to the correct path before running."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: Create venv and install ROCm SDK + build tools
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 1: Python venv + ROCm SDK"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

# Activate for pip installs
export PATH="$VENV_DIR/Scripts:$PATH"

pip install --quiet cmake ninja

# Install ROCm SDK via pip (nightly)
# Install device packages for ALL specified GPU targets
ROCM_PIP_ARGS=(--quiet --pre rocm_sdk_core rocm_sdk_devel rocm_sdk_libraries)
IFS=';' read -ra TARGETS_ARRAY <<< "$GPU_TARGETS"
for target in "${TARGETS_ARRAY[@]}"; do
    ROCM_PIP_ARGS+=("rocm_sdk_device_${target}")
done
pip install "${ROCM_PIP_ARGS[@]}" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    --extra-index-url https://pypi.org/simple/ \
    2>/dev/null || echo "WARNING: Some ROCm pip packages may have failed. Check manually."

# Set tool paths
CMAKE="$VENV_DIR/Scripts/cmake.exe"
NINJA="$VENV_DIR/Scripts/ninja.exe"
ROCM_DEVEL="$VENV_DIR/Lib/site-packages/_rocm_sdk_devel"
ROCM_CORE="$VENV_DIR/Lib/site-packages/_rocm_sdk_core"
ROCM_LIBS="$VENV_DIR/Lib/site-packages/_rocm_sdk_libraries"
CLANG_CL="$ROCM_DEVEL/lib/llvm/bin/clang-cl.exe"
CLANGXX="$ROCM_DEVEL/lib/llvm/bin/clang++.exe"

if [ ! -f "$CLANG_CL" ]; then
    echo "ERROR: clang-cl.exe not found at $CLANG_CL"
    echo "ROCm SDK devel package may not be installed correctly."
    exit 1
fi

export PATH="$VENV_DIR/Scripts:$ROCM_DEVEL/lib/llvm/bin:$ROCM_DEVEL/bin:$ROCM_CORE/bin:$WIN_SDK_BIN:$PATH"

echo "  CMake:    $("$CMAKE" --version | head -1)"
echo "  Ninja:    $("$NINJA" --version)"
echo "  Clang:    $("$CLANG_CL" --version | head -1)"

# ---------------------------------------------------------------------------
# Step 2: Clone source repos
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2: Clone source repos"

git config --global core.longpaths true

if [ ! -d "$MIGRAPHX_SRC" ]; then
    echo "  Cloning AMDMIGraphX..."
    git clone https://github.com/ROCm/AMDMIGraphX.git "$MIGRAPHX_SRC"
else
    echo "  AMDMIGraphX already exists, skipping clone."
fi

if [ ! -d "$CK_SRC" ]; then
    echo "  Cloning composable_kernel at $CK_COMMIT..."
    git clone https://github.com/ROCm/composable_kernel.git "$CK_SRC"
    (cd "$CK_SRC" && git checkout "$CK_COMMIT")
else
    echo "  composable_kernel already exists, skipping clone."
fi

if [ ! -d "$ROCMLIR_SRC" ]; then
    echo "  Cloning rocMLIR at $ROCMLIR_COMMIT..."
    git clone https://github.com/ROCm/rocMLIR.git "$ROCMLIR_SRC"
    (cd "$ROCMLIR_SRC" && git checkout "$ROCMLIR_COMMIT" && git submodule update --init --recursive)
else
    echo "  rocMLIR already exists, skipping clone."
fi

# ---------------------------------------------------------------------------
# Step 3: Apply Windows patches
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 3: Apply Windows patches"

# --- Patch 3a: CK codegen CMakeLists.txt ---
# clang-cl needs /std:c++20 not -std=c++20, and compile options must come
# before add_embed_library so the embed lib gets C++20 too.
CK_CMAKE="$CK_SRC/codegen/CMakeLists.txt"
if ! grep -q '/std:c++20' "$CK_CMAKE" 2>/dev/null; then
    echo "  Patching CK codegen CMakeLists.txt (C++20 for clang-cl)..."
    sed -i '/^include(Embed)/a\
\
if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))\
    add_compile_options(/std:c++20)\
else()\
    add_compile_options(-std=c++20)\
endif()' "$CK_CMAKE"
else
    echo "  CK codegen CMakeLists.txt already patched."
fi

# --- Patch 3b: CK utils.hpp missing #include <string> ---
CK_UTILS="$CK_SRC/codegen/include/ck/host/utils.hpp"
if ! grep -q '#include <string>' "$CK_UTILS" 2>/dev/null; then
    echo "  Patching CK utils.hpp (add #include <string>)..."
    sed -i '/#include <cstdint>/a #include <string>' "$CK_UTILS"
else
    echo "  CK utils.hpp already patched."
fi

# --- Patch 3c: MIGraphX ck.hpp — remove #ifndef _WIN32 guard on env vars ---
CK_HPP="$MIGRAPHX_SRC/src/targets/gpu/include/migraphx/gpu/ck.hpp"
if grep -q '#ifndef _WIN32' "$CK_HPP" 2>/dev/null; then
    echo "  Patching MIGraphX ck.hpp (remove _WIN32 guard on env vars)..."
    sed -i '/#ifndef _WIN32/{
        N;N;N;N;N
        s/#ifndef _WIN32\n\(MIGRAPHX_DECLARE_ENV_VAR.*\n\)\(MIGRAPHX_DECLARE_ENV_VAR.*\n\)\(MIGRAPHX_DECLARE_ENV_VAR.*\n\)\(MIGRAPHX_DECLARE_ENV_VAR.*\n\)#endif/\1\2\3\4/
    }' "$CK_HPP"
    # Fallback: if sed multiline didn't work, use python
    if grep -q '#ifndef _WIN32' "$CK_HPP" 2>/dev/null; then
        python -c "
import re
with open('$CK_HPP', 'r') as f:
    content = f.read()
content = content.replace('#ifndef _WIN32\nMIGRAPHX_DECLARE_ENV_VAR', 'MIGRAPHX_DECLARE_ENV_VAR')
content = re.sub(r'(MIGRAPHX_DECLARE_ENV_VAR\(MIGRAPHX_TUNE_CK\);)\s*#endif', r'\1', content)
with open('$CK_HPP', 'w') as f:
    f.write(content)
"
    fi
else
    echo "  MIGraphX ck.hpp already patched."
fi

# --- Patch 3d: MIGraphX PythonModules.cmake — skip non-numeric py.exe entries ---
# `py -0p` may list Astral/uv standalone pythons as "-V:Astral/CPython3.13.13 <path>"
# which the version regex can't parse -> list index out of range.
PY_MODS="$MIGRAPHX_SRC/cmake/PythonModules.cmake"
if [ -f "$PY_MODS" ] && ! grep -q 'MATCHES "\^-V:\[0-9\]"' "$PY_MODS" 2>/dev/null; then
    echo "  Patching PythonModules.cmake (skip non-numeric py entries)..."
    python -c "
with open(r'$PY_MODS','r') as f: c=f.read()
old='if(NOT _found_python MATCHES \"^\\\\\\\\*[ \\\\t]*\")'
new='if(NOT _found_python MATCHES \"^\\\\\\\\*[ \\\\t]*\" AND _found_python MATCHES \"^-V:[0-9]\")'
c=c.replace(old,new)
with open(r'$PY_MODS','w') as f: f.write(c)
"
fi

echo "  Patches applied."

# ---------------------------------------------------------------------------
# Step 4: Build C++ dependencies
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 4: Build C++ dependencies"

mkdir -p "$CGET_PREFIX"

CMAKE_GLOBAL_ARGS=(
    -G Ninja
    -DCMAKE_MAKE_PROGRAM="$NINJA"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER="$CLANG_CL"
    -DCMAKE_CXX_COMPILER="$CLANG_CL"
    -DCMAKE_INSTALL_PREFIX="$CGET_PREFIX"
    -DCMAKE_PREFIX_PATH="$CGET_PREFIX;$ROCM_DEVEL;$ROCM_CORE;$ROCM_LIBS"
    -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

build_dep() {
    local name="$1"; shift
    local src_dir="$1"; shift
    local build_dir="$src_dir/build"

    if [ -f "$build_dir/.build_done" ]; then
        echo "  [$name] Already built, skipping."
        return 0
    fi

    echo "  [$name] Configuring..."
    mkdir -p "$build_dir"
    "$CMAKE" "${CMAKE_GLOBAL_ARGS[@]}" "$@" -B "$build_dir" -S "$src_dir"

    echo "  [$name] Building..."
    # BUILD_TARGET (optional): build just one target, skipping test exes that
    # don't compile on Windows. Cleared after use by the caller.
    if [ -n "${BUILD_TARGET:-}" ]; then
        "$CMAKE" --build "$build_dir" --parallel "$JOBS" --target "$BUILD_TARGET"
    else
        "$CMAKE" --build "$build_dir" --parallel "$JOBS"
    fi

    echo "  [$name] Installing..."
    "$CMAKE" --install "$build_dir"

    touch "$build_dir/.build_done"
    echo "  [$name] Done."
}

clone_and_build_dep() {
    local name="$1"
    local repo="$2"
    local ref="$3"
    shift 3
    local src_dir="$ROOT_DIR/_deps/$name"

    # Already built? skip clone entirely
    if [ -f "$src_dir/build/.build_done" ]; then
        echo "  [$name] Already built, skipping."
        return 0
    fi

    if [ ! -d "$src_dir/.git" ]; then
        echo "  [$name] Cloning $repo @ $ref..."
        mkdir -p "$ROOT_DIR/_deps"
        # Stale partial dir (no .git) from an interrupted run? wipe first.
        [ -d "$src_dir" ] && rm -rf "$src_dir"
        # $ref may be a tag/branch or a 40-char SHA. --branch only takes tag/branch.
        if [[ "$ref" =~ ^[0-9a-fA-F]{40}$ ]]; then
            git clone "https://github.com/$repo.git" "$src_dir" && (cd "$src_dir" && git checkout "$ref")
        else
            git clone --depth 1 --branch "$ref" "https://github.com/$repo.git" "$src_dir" 2>/dev/null || \
            (git clone "https://github.com/$repo.git" "$src_dir" && cd "$src_dir" && git checkout "$ref")
        fi
    fi

    build_dep "$name" "$src_dir" "$@"
}

download_and_build_dep() {
    local name="$1"
    local url="$2"
    shift 2
    local src_dir="$ROOT_DIR/_deps/$name"

    # Re-extract if missing OR a prior run left an empty/partial dir.
    if [ ! -f "$src_dir/CMakeLists.txt" ]; then
        echo "  [$name] Downloading $url..."
        mkdir -p "$ROOT_DIR/_deps"
        [ -d "$src_dir" ] && rm -rf "$src_dir"
        local archive="$ROOT_DIR/_deps/${name}.tar.gz"
        curl -sL "$url" -o "$archive"
        mkdir -p "$src_dir"
        tar --force-local -xzf "$archive" -C "$src_dir" --strip-components=1
        rm -f "$archive"
    fi

    build_dep "$name" "$src_dir" "$@"
}

# 4a: rocm-cmake (build tools, needed by CK and MIGraphX)
echo ""
echo "  --- rocm-cmake ---"
clone_and_build_dep "rocm-cmake" "ROCm/rocm-cmake" "b3959201846b9d9250a6f2a386ad43be04d1d051"

# 4b: ROCm recipes (provides sqlite3 cmake recipe)
echo ""
echo "  --- rocm-recipes ---"
if [ ! -d "$ROOT_DIR/_deps/rocm-recipes" ]; then
    echo "  [rocm-recipes] Cloning..."
    git clone --depth 1 https://github.com/ROCm/rocm-recipes.git "$ROOT_DIR/_deps/rocm-recipes"
fi
pip install --quiet "$ROOT_DIR/_deps/rocm-recipes" 2>/dev/null || true

# 4c: abseil
echo ""
echo "  --- abseil ---"
# Pin ABSL_OPTION_USE_STD_STRING_VIEW to a deterministic value (default 2 =
# auto-detect produced a header/lib split: libs used std::string_view but the
# installed header stayed 0=absl's own class, so protobuf compiled against the
# wrong ABI -> migraphx_onnx.dll "undefined symbol absl::...string_view"). Force 1
# so header+libs+protobuf agree. Clone first so the source file exists to patch.
ABSEIL_SRC="$ROOT_DIR/_deps/abseil"
if [ ! -d "$ABSEIL_SRC/.git" ]; then
    [ -d "$ABSEIL_SRC" ] && rm -rf "$ABSEIL_SRC"
    git clone --depth 1 --branch 20250512.0 https://github.com/abseil/abseil-cpp.git "$ABSEIL_SRC"
fi
if [ -f "$ABSEIL_SRC/absl/base/options.h" ]; then
    echo "  Pinning ABSL_OPTION_USE_STD_STRING_VIEW=1..."
    sed -i 's/#define ABSL_OPTION_USE_STD_STRING_VIEW [0-9]/#define ABSL_OPTION_USE_STD_STRING_VIEW 1/' "$ABSEIL_SRC/absl/base/options.h"
fi
clone_and_build_dep "abseil" "abseil/abseil-cpp" "20250512.0" \
    -DABSL_ENABLE_INSTALL=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DABSL_MSVC_STATIC_RUNTIME=Off

# 4d: protobuf (needs rc.exe in PATH)
echo ""
echo "  --- protobuf ---"
if ! clone_and_build_dep "protobuf" "google/protobuf" "v30.0" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=On \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -Dprotobuf_BUILD_TESTS=Off \
    -Dprotobuf_BUILD_SHARED_LIBS=Off \
    -Dprotobuf_MSVC_STATIC_RUNTIME=Off \
    -Dprotobuf_BUILD_LIBPROTOC=On \
    -Dprotobuf_BUILD_PROTOC_BINARIES=On \
    -Dprotobuf_BUILD_PROTOBUF_BINARIES=On \
    -Dprotobuf_BUILD_LIBUPB=Off \
    -Dprotobuf_ABSL_PROVIDER=package 2>&1; then
    echo "  protobuf build failed. Retrying without protoc + downloading pre-built..."
    rm -rf "$ROOT_DIR/_deps/protobuf/build"
    clone_and_build_dep "protobuf" "google/protobuf" "v30.0" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=On \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -Dprotobuf_BUILD_TESTS=Off \
        -Dprotobuf_BUILD_SHARED_LIBS=Off \
        -Dprotobuf_MSVC_STATIC_RUNTIME=Off \
        -Dprotobuf_BUILD_LIBPROTOC=On \
        -Dprotobuf_BUILD_PROTOC_BINARIES=Off \
        -Dprotobuf_BUILD_PROTOBUF_BINARIES=On \
        -Dprotobuf_BUILD_LIBUPB=Off \
        -Dprotobuf_ABSL_PROVIDER=package
    echo "  Downloading pre-built protoc..."
    curl -sL "https://github.com/protocolbuffers/protobuf/releases/download/v30.0/protoc-30.0-win64.zip" \
        -o "$ROOT_DIR/_deps/protoc-win64.zip"
    (cd "$CGET_PREFIX" && unzip -o "$ROOT_DIR/_deps/protoc-win64.zip" "bin/*")
    rm -f "$ROOT_DIR/_deps/protoc-win64.zip"
    # libprotobuf-only install still declares protobuf::protoc with a blank
    # IMPORTED_LOCATION -> MIGraphX protobuf_generate emits malformed
    # "protoc --cpp_out :path". Point the imported target at the prebuilt exe.
    PROTOC_EXE="$CGET_PREFIX/bin/protoc.exe"
    PB_CMAKE_DIR="$CGET_PREFIX/lib/cmake/protobuf"
    if [ -d "$PB_CMAKE_DIR" ]; then
        cat > "$PB_CMAKE_DIR/protoc-prebuilt-fix.cmake" << FIXEOF
if(NOT TARGET protobuf::protoc)
  add_executable(protobuf::protoc IMPORTED)
endif()
set_property(TARGET protobuf::protoc PROPERTY IMPORTED_LOCATION "$PROTOC_EXE")
FIXEOF
        # also rewrite any blank IMPORTED_LOCATION on the protoc target
        python -c "
import glob,re,os
for f in glob.glob(os.path.join(r'$PB_CMAKE_DIR','*.cmake')):
    with open(f) as fh: c=fh.read()
    if 'protobuf::protoc' in c:
        c=re.sub(r'(set_target_properties\(protobuf::protoc PROPERTIES[^)]*IMPORTED_LOCATION(?:_[A-Z]+)?\s+)\"[^\"]*\"', r'\1\"$PROTOC_EXE\"', c)
        with open(f,'w') as fh: fh.write(c)
" 2>/dev/null || true
    fi
    echo "  Pre-built protoc installed + cmake target patched at $CGET_PREFIX/bin/"
fi

# 4e: nlohmann/json
echo ""
echo "  --- nlohmann/json ---"
clone_and_build_dep "json" "nlohmann/json" "v3.8.0" \
    -DJSON_BuildTests=Off

# 4f: pybind11
echo ""
echo "  --- pybind11 ---"
clone_and_build_dep "pybind11" "pybind/pybind11" "3e9dfa2866941655c56877882565e7577de6fc7b" \
    -DPYBIND11_TEST=Off

# 4g: msgpack-c
echo ""
echo "  --- msgpack-c ---"
clone_and_build_dep "msgpack-c" "msgpack/msgpack-c" "cpp-3.3.0" \
    -DMSGPACK_BUILD_TESTS=Off \
    -DMSGPACK_BUILD_EXAMPLES=Off

# 4h: sqlite3 (download amalgamation + compile directly, copy to cget)
echo ""
echo "  --- sqlite3 ---"
SQLITE_SRC="$ROOT_DIR/_deps/sqlite3"
SQLITE_LIB="$CGET_PREFIX/lib/sqlite3.lib"
SQLITE_HDR="$CGET_PREFIX/include/sqlite3.h"
if [ -f "$SQLITE_LIB" ] && [ -f "$SQLITE_HDR" ]; then
    echo "  [sqlite3] Already installed, skipping."
else
    if [ ! -f "$SQLITE_SRC/sqlite3.c" ]; then
        echo "  [sqlite3] Downloading amalgamation..."
        rm -rf "$SQLITE_SRC"
        mkdir -p "$ROOT_DIR/_deps"
        curl -sL "https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip" -o "$ROOT_DIR/_deps/sqlite3.zip"
        (cd "$ROOT_DIR/_deps" && unzip -q sqlite3.zip && mv sqlite-amalgamation-3500400 sqlite3)
        rm -f "$ROOT_DIR/_deps/sqlite3.zip"
    fi
    echo "  [sqlite3] Compiling sqlite3.c -> sqlite3.lib..."
    mkdir -p "$CGET_PREFIX/lib" "$CGET_PREFIX/include"
    LLVM_LIB="$ROCM_DEVEL/lib/llvm/bin/llvm-lib.exe"
    (cd "$SQLITE_SRC" && \
        "$CLANG_CL" /nologo /c /O2 /MD /DSQLITE_ENABLE_COLUMN_METADATA sqlite3.c /Fosqlite3.obj && \
        "$LLVM_LIB" /nologo /out:sqlite3.lib sqlite3.obj && \
        cp sqlite3.lib "$SQLITE_LIB" && \
        cp sqlite3.h "$SQLITE_HDR" && \
        { [ -f sqlite3ext.h ] && cp sqlite3ext.h "$CGET_PREFIX/include/sqlite3ext.h" || true; })
    if [ ! -f "$SQLITE_LIB" ] || [ ! -f "$SQLITE_HDR" ]; then
        echo "ERROR: sqlite3 install failed - missing lib or header"
        exit 1
    fi
    echo "  [sqlite3] Installed sqlite3.lib + sqlite3.h to $CGET_PREFIX"
fi

# 4i: eigen
echo ""
echo "  --- eigen ---"
download_and_build_dep "eigen" \
    "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz" \
    -DBUILD_TESTING=Off \
    -DEIGEN_BUILD_DOC=Off \
    -DEIGEN_BUILD_LAPACK=Off

# 4j: Composable Kernel (codegen only)
echo ""
echo "  --- Composable Kernel (codegen) ---"
BUILD_TARGET=ck_host build_dep "ck-codegen" "$CK_SRC/codegen" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=On \
    -DBUILD_TESTING=Off \
    -DEMBED_USE=CArrays

# 4k: rocMLIR (full LLVM/MLIR build — SLOW, 1-3 hours)
echo ""
echo "  --- rocMLIR (this takes 1-3 hours) ---"
build_dep "rocMLIR" "$ROCMLIR_SRC" \
    -DBUILD_FAT_LIBROCKCOMPILER=On \
    -DLLVM_INCLUDE_TESTS=Off \
    -DLLVM_DISABLE_ASSEMBLY_FILES=ON \
    -DBUILD_SHARED_LIBS=OFF

echo ""
echo ">>> All dependencies built."

# ---------------------------------------------------------------------------
# Step 5: Build MIGraphX
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 5: Build MIGraphX"

MIGRAPHX_BUILD="$MIGRAPHX_SRC/build"
mkdir -p "$MIGRAPHX_BUILD"

if [ ! -f "$MIGRAPHX_BUILD/.configure_done" ]; then
    # Stale CMakeCache.txt (different GPU targets, cached *-NOTFOUND) poisons
    # this run. Start clean.
    if [ -f "$MIGRAPHX_BUILD/CMakeCache.txt" ]; then
        echo "  Removing stale CMakeCache.txt for clean configure..."
        rm -f "$MIGRAPHX_BUILD/CMakeCache.txt"
        rm -rf "$MIGRAPHX_BUILD/CMakeFiles"
    fi
    # protobuf is libprotobuf-only (absl as package) -> libprotobuf.lib has
    # unresolved abseil symbols. MIGraphX links the bare Protobuf_LIBRARY path and
    # drops protobuf's transitive absl -> migraphx_onnx.dll "undefined symbol absl::".
    # Fix: Protobuf_LIBRARY = libprotobuf + every absl/utf8 static lib (lld-link
    # pulls only referenced symbols, so over-listing is harmless).
    PROTOBUF_LIB_LIST="$CGET_PREFIX/lib/libprotobuf.lib"
    for absl in "$CGET_PREFIX"/lib/absl_*.lib; do
        [ -f "$absl" ] && PROTOBUF_LIB_LIST="$PROTOBUF_LIB_LIST;$absl"
    done
    for extra in utf8_range.lib utf8_validity.lib; do
        [ -f "$CGET_PREFIX/lib/$extra" ] && PROTOBUF_LIB_LIST="$PROTOBUF_LIB_LIST;$CGET_PREFIX/lib/$extra"
    done
    echo "  Configuring MIGraphX..."
    "$CMAKE" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CLANG_CL" \
        -DCMAKE_CXX_COMPILER="$CLANG_CL" \
        -DCMAKE_HIP_COMPILER="$CLANGXX" \
        -DCMAKE_HIP_ARCHITECTURES="$GPU_TARGETS" \
        -DCMAKE_PREFIX_PATH="$CGET_PREFIX;$ROCM_DEVEL;$ROCM_CORE;$ROCM_LIBS" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_POLICY_DEFAULT_CMP0091=NEW \
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL \
        -DMIGRAPHX_ENABLE_GPU=ON \
        -DMIGRAPHX_ENABLE_CPU=OFF \
        -DMIGRAPHX_ENABLE_FPGA=OFF \
        -DMIGRAPHX_ENABLE_PYTHON=ON \
        -DMIGRAPHX_USE_COMPOSABLEKERNEL=ON \
        -DMIGRAPHX_USE_ROCBLAS=ON \
        -DMIGRAPHX_USE_HIPBLASLT=ON \
        -DMIGRAPHX_USE_MIOPEN=ON \
        -DMIGRAPHX_ENABLE_MLIR=ON \
        -DMIGRAPHX_WORKAROUND_HIP_MULTI_ARCH_BUG=ON \
        -DGPU_TARGETS="$GPU_TARGETS" \
        -DBUILD_DEV=OFF \
        -DPython3_EXECUTABLE="$VENV_DIR/Scripts/python.exe" \
        -DPYTHON_EXECUTABLE="$VENV_DIR/Scripts/python.exe" \
        -DSQLite3_ROOT="$CGET_PREFIX" \
        -DSQLite3_INCLUDE_DIR="$CGET_PREFIX/include" \
        -DSQLite3_LIBRARY="$CGET_PREFIX/lib/sqlite3.lib" \
        -Dmsgpackc-cxx_DIR="$CGET_PREFIX/lib/cmake/msgpack" \
        -DProtobuf_INCLUDE_DIR="$CGET_PREFIX/include" \
        -DProtobuf_LIBRARY="$PROTOBUF_LIB_LIST" \
        -DProtobuf_PROTOC_EXECUTABLE="$CGET_PREFIX/bin/protoc.exe" \
        -B "$MIGRAPHX_BUILD" \
        -S "$MIGRAPHX_SRC"

    touch "$MIGRAPHX_BUILD/.configure_done"
fi

echo "  Building MIGraphX (this may take 30-90 minutes)..."
"$CMAKE" --build "$MIGRAPHX_BUILD" --parallel "$JOBS"

echo "  Installing MIGraphX..."
"$CMAKE" --install "$MIGRAPHX_BUILD"

echo ""
echo ">>> MIGraphX build complete!"

# ---------------------------------------------------------------------------
# Step 6: Verify
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 6: Quick verification"

MIGRAPHX_PYTHON="$MIGRAPHX_BUILD/lib"
MIGRAPHX_LIB="$INSTALL_DIR/lib"
MIGRAPHX_BIN="$INSTALL_DIR/bin"

echo "  Libraries:"
ls "$INSTALL_DIR/lib/"migraphx*.dll 2>/dev/null || ls "$INSTALL_DIR/bin/"migraphx*.dll 2>/dev/null || echo "  (checking build dir)" && ls "$MIGRAPHX_BUILD/lib/"migraphx*.dll 2>/dev/null || true

echo ""
echo "  Python module:"
ls "$MIGRAPHX_BUILD/lib/migraphx"*.pyd 2>/dev/null || ls "$MIGRAPHX_BUILD/lib/"*migraphx*.pyd 2>/dev/null || echo "  (will check after install)"

echo ""
echo "============================================"
echo "BUILD COMPLETE"
echo "============================================"
echo ""
echo "GPU targets:  $GPU_TARGETS"
echo "Install dir:  $INSTALL_DIR"
echo "Build dir:    $MIGRAPHX_BUILD"
echo ""
echo "To test Python bindings:"
echo "  export PATH=\"$INSTALL_DIR/lib:\$PATH\""
echo "  export PYTHONPATH=\"$MIGRAPHX_BUILD/lib:\$PYTHONPATH\""
echo "  python -c \"import migraphx; print(migraphx)\""
echo ""
echo "Known patches applied:"
echo "  1. CK codegen: /std:c++20 for clang-cl + placed before add_embed_library"
echo "  2. CK utils.hpp: added #include <string>"
echo "  3. CK codegen: EMBED_USE=CArrays (avoids RC long path issues)"
echo "  4. rocMLIR: LLVM_DISABLE_ASSEMBLY_FILES=ON (avoids MASM/clang flag conflicts)"
echo "  5. MIGraphX ck.hpp: removed #ifndef _WIN32 guard on env var declarations"
echo "  6. git config core.longpaths=true (CK repo has long filenames)"
