#################################################################################
# Script for generating the kero wheel.
# ```
# $ python setup.py bdist_wheel
# ```
# Environment variables:
#
#   THIRDPARTY_LLVM_DIR:
#       Path to the llvm-project
#
#   LLVM_INSTALL_DIR:
#       Path to the install directory for llvm-project
#
#   KERO_BUILD_DIR:
#        Path to kero build directory
#
#   CMAKE_BUILD_TYPE:
#       Release or Debug or other types
#
#   BUILD_SYSTEM:
#       Which build system to use Default is Ninja
#
#   MAX_JOBS:
#       No. of cores to use
#
#################################################################################

import os
import sys
import shutil
import pathlib
import subprocess
import multiprocessing

from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

def get_kero_version(official_version=False):
    version = pathlib.Path(os.path.dirname(__file__), "version.txt").read_text().strip()
    if official_version:
        return version

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
    )
    git_sha = revision.decode("ascii").strip()

    return version + "+git" + git_sha[:7]

SETUPPY_DIR = os.path.realpath(os.path.dirname(__file__))

THIRDPARTY_LLVM_DIR = os.getenv("THIRDPARTY_LLVM_DIR", None)
LLVM_INSTALL_DIR = os.getenv("LLVM_INSTALL_DIR", None)
KERO_BUILD_DIR = os.getenv("KERO_BUILD_DIR", os.path.join(SETUPPY_DIR, "build"))
CMAKE_BUILD_TYPE = os.getenv("CMAKE_BUILD_TYPE", "Release")
BUILD_SYSTEM = os.getenv("BUILD_SYSTEM", "Ninja")
MAX_JOBS = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))

official_version = False
if CMAKE_BUILD_TYPE == "Release":
    official_version = True

PACKAGE_VERSION = get_kero_version(official_version)

CMAKE_INSTALL_DIR_REL = os.path.join("build", "setup_install")
CMAKE_INSTALL_DIR_ABS = os.path.join(SETUPPY_DIR, CMAKE_INSTALL_DIR_REL)

def prepare_installation():
    if THIRDPARTY_LLVM_DIR is None and LLVM_INSTALL_DIR is None:
        print("Set either THIRDPARTY_LLVM_DIR or LLVM_INSTALL_DIR first")
        sys.exit(1)

    os.makedirs(KERO_BUILD_DIR, exist_ok=True)

    cmake_config_args = [
        "cmake",
        f"-G {BUILD_SYSTEM}",
        f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
        f"-DPython3_EXECUTABLE={sys.executable}",
        "-DPython3_FIND_VIRTUALENV=ONLY",
        f"-DPython_EXECUTABLE={sys.executable}",
        "-DPython_FIND_VIRTUALENV=ONLY",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=kero",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DLLVM_ENABLE_ZSTD=OFF",
        "-DCMAKE_PLATFORM_NO_VERSIONED_SONAME=ON",
        "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",
        "-DCMAKE_C_VISIBILITY_PRESET=hidden",
        "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
        SETUPPY_DIR,
    ]
    
    if LLVM_INSTALL_DIR:
        cmake_config_args += [
            f"-DMLIR_DIR={LLVM_INSTALL_DIR}/lib/cmake/mlir/",
            f"-DLLVM_DIR={LLVM_INSTALL_DIR}/lib/cmake/llvm/",
        ]
    else:
        cmake_config_args += [
            f"-DMLIR_DIR={THIRDPARTY_LLVM_DIR}/build/lib/cmake/mlir/",
            f"-DLLVM_DIR={THIRDPARTY_LLVM_DIR}/build/lib/cmake/llvm/",
        ]
        
    cmake_build_args = [
        "cmake",
        "--build",
        ".",
        "--config",
        CMAKE_BUILD_TYPE,
        "--target",
        "KeroEngineMLIRPythonModules",
        "--",
        f"-j{MAX_JOBS}",
    ]
    
    try:
        subprocess.check_call(cmake_config_args, cwd=KERO_BUILD_DIR)
        subprocess.check_call(cmake_build_args, cwd=KERO_BUILD_DIR)
    except subprocess.CalledProcessError as e:
        print(f"CMake execution failed: {e}")
        sys.exit(e.returncode)

    if os.path.exists(CMAKE_INSTALL_DIR_ABS):
        shutil.rmtree(CMAKE_INSTALL_DIR_ABS)
    os.makedirs(CMAKE_INSTALL_DIR_ABS, exist_ok=True)

    source_package_src = os.path.join(SETUPPY_DIR, "kero")
    source_package_dst = os.path.join(CMAKE_INSTALL_DIR_ABS, "kero")
    
    if os.path.exists(source_package_src):
        for root, dirs, files in os.walk(source_package_src):
            # Prune __pycache__ directories from the walk tree
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
                
            rel_path = os.path.relpath(root, source_package_src)
            target_subdir = source_package_dst if rel_path == "." else os.path.join(source_package_dst, rel_path)
            os.makedirs(target_subdir, exist_ok=True)
            for file in files:
                # Enforce exact match for source python files
                if file.endswith(".py"):
                    shutil.copy2(os.path.join(root, file), os.path.join(target_subdir, file))

    generated_package_src = os.path.join(KERO_BUILD_DIR, "python_packages", "kero")
    if os.path.exists(generated_package_src):
        for root, dirs, files in os.walk(generated_package_src):
            # Prune __pycache__ directories from the walk tree
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")
                
            rel_path = os.path.relpath(root, generated_package_src)
            target_subdir = source_package_dst if rel_path == "." else os.path.join(source_package_dst, rel_path)
            os.makedirs(target_subdir, exist_ok=True)
            for file in files:
                # Structural fix: split file extension checks to prevent tracking .pyc files
                if file.endswith(".py") or file.endswith((".so", ".pyd", ".dylib")):
                    s_file = os.path.join(root, file)
                    d_file = os.path.join(target_subdir, file)
                    if os.path.exists(d_file):
                        os.remove(d_file)
                    shutil.copy2(s_file, d_file)


prepare_installation()

discovered_packages = find_namespace_packages(where=CMAKE_INSTALL_DIR_ABS)

class CMakeBuildPy(_build_py):
    def run(self):
        target_dir = os.path.abspath(self.build_lib)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(
            CMAKE_INSTALL_DIR_ABS,
            target_dir,
            symlinks=self.editable_mode,
            dirs_exist_ok=True,
        )


class CustomBuild(_build):
    def run(self):
        self.run_command("build_py")
        self.run_command("build_ext")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", **kwargs):
        Extension.__init__(self, name, sources=[], **kwargs)
        self.sourcedir = os.path.abspath(sourcedir)


class NoopBuildExtension(_build_ext):
    def build_extension(self, ext):
        pass


_bdist_wheel_cmdclass = {}
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def get_tag(self):
            python, abi, plat = _bdist_wheel.get_tag(self)
            return python, abi, plat

    _bdist_wheel_cmdclass = {"bdist_wheel": bdist_wheel}
except ImportError:
    pass

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kero",
    version=PACKAGE_VERSION,
    author="PyDevC",
    author_email="pydevc@proton.me",
    description="GPU Accelerated SQL Query Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    cmdclass=dict(
        {
            "build": CustomBuild,
            "built_ext": NoopBuildExtension,
            "build_ext": NoopBuildExtension,
            "build_py": CMakeBuildPy,
        },
        **_bdist_wheel_cmdclass,
    ),
    ext_modules=[
        CMakeExtension("kero._engine._kero._mlir_libs._keroEngine"),
    ],
    zip_safe=False,
    package_dir={
        "": CMAKE_INSTALL_DIR_REL,
    },
    packages=discovered_packages,
    python_requires=">=3.9",
    install_requires=[
        "packaging",
        "numpy",
        "pyarrow",
        "sqlglot",
    ],
    extras_require={
        "cuda": ["torch>=2.5.0"],
        "rocm": ["torch>=2.5.0"],
        "torch": ["torch>=2.5.0"],
    },
)
