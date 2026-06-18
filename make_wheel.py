#!/usr/bin/env python3
"""Build migraphx_rocm-*.whl from build_gpu/bin output.

Wheel contains only migraphx binaries + a .pth setup file.
ROCm SDK DLLs (amdhip64, hiprtc, rocblas, MIOpen, etc.) come from
the pip packages _rocm_sdk_core and _rocm_sdk_libraries, which are
declared as prerequisites.  The .pth file adds those packages' bin dirs
to os.add_dll_directory (Python process) and PATH (subprocess).
"""
import os, sys, zipfile, hashlib, base64
from pathlib import Path

BUILD_DIR = Path(r"F:\MIGraphxWin\build_gpu\bin")
DIST_DIR  = Path(r"F:\MIGraphxWin\dist")
VERSION   = "2.16.0.dev0"
WHEEL_TAG = "cp312-cp312-win_amd64"
PKG_NAME  = "migraphx_rocm"
WHEEL_STEM = f"{PKG_NAME}-{VERSION}-{WHEEL_TAG}"
DIST_INFO  = f"{PKG_NAME}-{VERSION}.dist-info"

# --- migraphx binaries from build_gpu/bin ---
MIGRAPHX_FILES = [
    "migraphx.cp312-win_amd64.pyd",
    "migraphx.dll",
    "migraphx_device.dll",
    "migraphx_gpu.dll",
    "migraphx_onnx.dll",
    "migraphx_tf.dll",
    "migraphx_py.dll",
    "migraphx_py_3.12.dll",
    "migraphx_ref.dll",
    "migraphx-hiprtc-driver.exe",
    "sqlite3.dll",
]

# GPU device packages — provide rocBLAS/hipBLASLt kernel databases per arch
DEVICE_PACKAGES = [
    "rocm-sdk-device-gfx900",
    "rocm-sdk-device-gfx906",
    "rocm-sdk-device-gfx908",
    "rocm-sdk-device-gfx90a",
    "rocm-sdk-device-gfx942",
    "rocm-sdk-device-gfx950",
    "rocm-sdk-device-gfx1010",
    "rocm-sdk-device-gfx1011",
    "rocm-sdk-device-gfx1012",
    "rocm-sdk-device-gfx1030",
    "rocm-sdk-device-gfx1031",
    "rocm-sdk-device-gfx1032",
    "rocm-sdk-device-gfx1033",
    "rocm-sdk-device-gfx1034",
    "rocm-sdk-device-gfx1035",
    "rocm-sdk-device-gfx1036",
    "rocm-sdk-device-gfx1100",
    "rocm-sdk-device-gfx1101",
    "rocm-sdk-device-gfx1102",
    "rocm-sdk-device-gfx1103",
    "rocm-sdk-device-gfx1150",
    "rocm-sdk-device-gfx1151",
    "rocm-sdk-device-gfx1152",
    "rocm-sdk-device-gfx1153",
    "rocm-sdk-device-gfx1200",
    "rocm-sdk-device-gfx1201",
    "rocm-sdk-device-gfx1250",
]

# Optional extras: one per arch + "all" umbrella
_EXTRA_LINES = []
for p in DEVICE_PACKAGES:
    arch = p.replace("rocm-sdk-device-", "")
    _EXTRA_LINES.append(f"Provides-Extra: {arch}")
    _EXTRA_LINES.append(f'Requires-Dist: {p}; extra == "{arch}"')
# "all" extra pulls everything
_EXTRA_LINES.append("Provides-Extra: all")
for p in DEVICE_PACKAGES:
    arch = p.replace("rocm-sdk-device-", "")
    _EXTRA_LINES.append(f'Requires-Dist: {p}; extra == "all"')
_EXTRAS_BLOCK = "\n".join(_EXTRA_LINES)

# --- generated file contents ---

METADATA_CONTENT = f"""\
Metadata-Version: 2.1
Name: {PKG_NAME}
Version: {VERSION}
Summary: MIGraphX GPU inference library for AMD ROCm 7.x (Windows)
Home-page: https://github.com/ROCm/AMDMIGraphX
Author: AMD ROCm
License: MIT
Requires-Python: >=3.10
Classifier: Programming Language :: Python :: 3
Classifier: License :: OSI Approved :: MIT License
Classifier: Operating System :: Microsoft :: Windows
{_EXTRAS_BLOCK}
Description-Content-Type: text/plain

MIGraphX 2.16.0.dev built for AMD GPUs on Windows with ROCm 7.14.
Supports MIOpen, rocBLAS, hipBLASLt.  Targets: 27 GPU architectures
(gfx900-gfx1250 including RDNA 4 gfx1200/gfx1201 and MI300 gfx942/gfx950).

Prerequisites:
  pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \\
      "torch[device-gfx120X-all]"

Install (bare — no device packages):
  pip install migraphx_rocm-*.whl

Install with all GPU kernel databases:
  pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \\
      "migraphx_rocm-*.whl[all]"

Install for your specific GPU (e.g. gfx1200):
  pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \\
      "migraphx_rocm-*.whl[gfx1200,gfx1201]"

Usage:
  import migraphx  # .pth configures DLL paths at Python startup
  print(migraphx.__version__)
"""

WHEEL_CONTENT = f"""\
Wheel-Version: 1.0
Generator: migraphx-build
Root-Is-Purelib: false
Tag: {WHEEL_TAG}
"""

TOP_LEVEL_CONTENT = "migraphx\n"

# .pth file: executed at Python startup, adds ROCm SDK bin dirs to PATH + DLL dirs
PTH_CONTENT = "import migraphx_rocm_setup\n"

# setup module: adds _rocm_sdk_core/bin + _rocm_sdk_libraries/bin to PATH and DLL dirs
SETUP_CONTENT = '''\
"""Auto-configured at Python startup via migraphx_rocm.pth."""
import os as _os

def _setup():
    _sp = _os.path.dirname(_os.path.abspath(__file__))
    _bins = [
        _os.path.join(_sp, '_rocm_sdk_libraries', 'bin'),
        _os.path.join(_sp, '_rocm_sdk_core', 'bin'),
    ]
    for _d in _bins:
        if not _os.path.isdir(_d):
            continue
        # PATH is inherited by subprocesses (migraphx-hiprtc-driver.exe)
        _os.environ['PATH'] = _d + _os.pathsep + _os.environ.get('PATH', '')
        # os.add_dll_directory affects the current Python process only
        try:
            _os.add_dll_directory(_d)
        except (OSError, AttributeError):
            pass

_setup()
del _setup
'''


def sha256_record(data: bytes) -> str:
    h = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(h).rstrip(b"=").decode()


def build_wheel():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out = DIST_DIR / f"{WHEEL_STEM}.whl"

    records: list[tuple[str, str, int]] = []

    def add_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes):
        zf.writestr(arcname, data, compress_type=zipfile.ZIP_DEFLATED)
        records.append((arcname, sha256_record(data), len(data)))

    def add_file(zf: zipfile.ZipFile, arcname: str, src: Path):
        data = src.read_bytes()
        zf.write(str(src), arcname, compress_type=zipfile.ZIP_DEFLATED)
        records.append((arcname, sha256_record(data), len(data)))

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # generated files
        add_bytes(zf, "migraphx_rocm_setup.py", SETUP_CONTENT.encode())
        add_bytes(zf, "migraphx_rocm.pth",      PTH_CONTENT.encode())

        # migraphx binaries
        for name in MIGRAPHX_FILES:
            src = BUILD_DIR / name
            if src.exists():
                add_file(zf, name, src)
                print(f"  + {name}  ({src.stat().st_size // 1024} KB)")
            else:
                print(f"  - SKIP {name} (not found)")

        # dist-info metadata
        add_bytes(zf, f"{DIST_INFO}/METADATA",      METADATA_CONTENT.encode())
        add_bytes(zf, f"{DIST_INFO}/WHEEL",         WHEEL_CONTENT.encode())
        add_bytes(zf, f"{DIST_INFO}/top_level.txt", TOP_LEVEL_CONTENT.encode())

        # RECORD (self-referential — last entry has empty hash)
        record_lines = [f"{name},{h},{size}" for name, h, size in records]
        record_lines.append(f"{DIST_INFO}/RECORD,,")
        add_bytes(zf, f"{DIST_INFO}/RECORD",
                  "\n".join(record_lines).encode())

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWheel: {out}")
    print(f"Size:  {size_mb:.1f} MB")


if __name__ == "__main__":
    build_wheel()
