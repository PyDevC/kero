# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

**MLIR-based GPU Accelerated SQL Query Engine for Deep Learning**
KeroSine is an MLIR-based SQL compiler designed to integrate structured data workflows with deep learning pipelines. It functions as a high-performance DataLoader that converts SQL query results directly into tensors, leveraging GPU acceleration to optimize the entire SQL-to-tensor pipeline for frameworks such as PyTorch.

> **Note:** Dependencies, full API documentation, and additional docs will be updated at the first pre-release.

## Prerequisites

Before building this project, make sure you have all dependencies installed.

- CMake
- Ninja build system
- LLVM / MLIR (built from source — see instructions below)
- CUDA Toolkit (for NVIDIA GPU support) (Optional)
- ROCm Toolkit (for AMD GPU support) (Optional)

## Build Instructions

### Build LLVM from Source

```bash
git clone https://github.com/llvm/llvm-project.git

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir;clang-tools-extra;compiler-rt;llvm;lldb;" \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=3 \
  -DLLVM_PARALLEL_TABLEGEN_JOBS=5

ninja -C build
```

> **Tip:** LLVM compilation is resource-intensive. If your system has limited RAM, reduce `DLLVM_PARALLEL_{COMPILE,LINK}_JOBS` to `3` or `5` to avoid system crash.

### Build Kero-Sine

```bash
export THIRDPARTY_LLVM_DIR="/path/to/llvm-project"

bash scripts/py_build.sh
```
