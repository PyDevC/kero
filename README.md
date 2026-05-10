# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

Kero-Sine is a SQL Query Engine build using MLIR and PyTorch, that has capability to run and optimize queries on CPU and GPU.
It uses `SQLGlot` as it's user parser which can handle different kinds of queries from different `dialects`. We use Apache Arrow to interact with existing databases and apply ETL on them, utlizing `Arrow Column Format`.
It provides custom Python wrappers around PyTorch `torch.Tensor` to act as Table, Column, Date Entry, etc which can be used as result types.

> **Note:** Dependencies, full API documentation, and additional docs will be updated at the first release.

## Prerequisites

- CMake (Min. version 3.20)
- Ninja build system (1.13)
- LLVM / MLIR (built from source — see instructions below) (Primarily from LLVM version 23.0.0)

## Build Instructions

### Build LLVM from Source

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
  -DLLVM_USE_LINKER=lld \
  -DLLVM_PARALLEL_LINK_JOBS=3 \
  -DLLVM_PARALLEL_TABLEGEN_JOBS=5

ninja -C build install
```

> **Tip:** LLVM compilation is resource-intensive. If your system has limited RAM, reduce `DLLVM_PARALLEL_{COMPILE,LINK}_JOBS` to `3` or `5` to avoid system crash.

### Install/Build Kero-Sine from source

```bash
pip install -r requirements.txt
export THIRDPARTY_LLVM_DIR="/path/to/llvm-project"

bash scripts/py_build.sh
```
