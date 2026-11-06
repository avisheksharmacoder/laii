# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Avishek and Contributors
"""
setup.py - Automated C11 Shared Library compilation and Python packaging for LFM2.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


ROOT_DIR = Path(__file__).resolve().parent
SRC_C = ROOT_DIR / "src" / "lfm2_api.c"


def compile_liblfm2(target_path: Path) -> None:
    """Compiles src/lfm2_api.c into a shared library liblfm2.so with platform ISA flags."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    arch = platform.machine().lower()

    cflags = [
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        "-std=c11",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wno-unused-function",
        f"-I{ROOT_DIR / 'include'}",
        "-D_GNU_SOURCE",
    ]

    if arch in ("x86_64", "amd64"):
        cflags.extend(["-mavx2", "-mfma", "-mf16c"])
    elif "arm" in arch or "aarch64" in arch:
        cflags.append("-march=armv8.2-a+dotprod+fp16")

    ldflags = ["-lm", "-lpthread"]

    cmd = [cc] + cflags + [str(SRC_C), "-o", str(target_path)] + ldflags
    print(f"[setup.py] Compiling C11 engine: {' '.join(cmd)}")
    subprocess.check_call(cmd)


class LFM2BuildExt(build_ext):
    def run(self):
        super().run()
        py_target = ROOT_DIR / "python" / "liblfm2.so"
        lib_target = ROOT_DIR / "lib" / "liblfm2.so"
        compile_liblfm2(py_target)
        if not lib_target.exists():
            shutil.copy2(py_target, lib_target)

        if self.build_lib:
            pkg_dest = Path(self.build_lib) / "lfm2" / "liblfm2.so"
            pkg_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_target, pkg_dest)


class LFM2BuildPy(build_py):
    def run(self):
        py_target = ROOT_DIR / "python" / "liblfm2.so"
        lib_target = ROOT_DIR / "lib" / "liblfm2.so"
        if not py_target.exists():
            if lib_target.exists():
                shutil.copy2(lib_target, py_target)
            else:
                compile_liblfm2(py_target)

        super().run()

        if self.build_lib:
            pkg_dest = Path(self.build_lib) / "lfm2" / "liblfm2.so"
            pkg_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(py_target, pkg_dest)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class LFM2BdistWheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            # Force platform-specific wheel tag because liblfm2.so is native binary
            self.root_is_pure = False
except ImportError:
    LFM2BdistWheel = None


cmdclass = {
    "build_ext": LFM2BuildExt,
    "build_py": LFM2BuildPy,
}
if LFM2BdistWheel is not None:
    cmdclass["bdist_wheel"] = LFM2BdistWheel

setup(cmdclass=cmdclass)
