# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Avishek and Contributors
"""
p4a-recipes/lfm2/__init__.py - Custom python-for-android recipe for LFM2.c.
Builds liblfm2.so for Android arm64-v8a with 16KB page alignment.
"""

from pathlib import Path
import shutil
import sh
from pythonforandroid.recipe import Recipe
from pythonforandroid.util import current_directory


class LFM2Recipe(Recipe):
    name = "lfm2"
    version = "1.0.0"
    depends = ["python3"]
    built_libraries = {"liblfm2.so": "."}

    def get_recipe_dir(self):
        return str(Path(__file__).resolve().parent)

    def should_build(self, arch):
        lib_out = Path(self.get_build_dir(arch.arch)) / "liblfm2.so"
        return not lib_out.exists()

    def prebuild_arch(self, arch):
        super().prebuild_arch(arch)
        src_dir = Path(self.get_recipe_dir()) / "src"
        build_dir = Path(self.get_build_dir(arch.arch))
        if src_dir.exists():
            shutil.copytree(src_dir, build_dir, dirs_exist_ok=True)

    def build_arch(self, arch):
        env = self.get_recipe_env(arch)
        build_dir = Path(self.get_build_dir(arch.arch))

        cc_bin = env.get("CC", "clang").split()[0]
        cc = sh.Command(cc_bin)

        cflags = [
            "-std=c11",
            "-O3",
            "-fPIC",
            "-shared",
            "-fvisibility=hidden",
            "-pthread",
            "-Wall",
            "-I./include",
            "-D_GNU_SOURCE",
        ]

        if arch.arch == "arm64-v8a":
            cflags.append("-march=armv8.2-a+dotprod+fp16")

        ldflags = [
            "-Wl,-soname,liblfm2.so",
            "-Wl,-z,max-page-size=16384",
            "-lm",
            "-llog",
        ]

        with current_directory(str(build_dir)):
            cc(
                *cflags,
                "src/lfm2_api.c",
                "-o", "liblfm2.so",
                *ldflags,
                _env=env
            )

            # Copy pure-stdlib ctypes wrapper into Android Python site-packages
            site_pkgs = Path(self.ctx.get_site_packages_dir(arch))
            if (build_dir / "lfm2.py").exists():
                shutil.copy2(build_dir / "lfm2.py", site_pkgs / "lfm2.py")

        self.install_libs(arch, str(build_dir / "liblfm2.so"))


recipe = LFM2Recipe()

