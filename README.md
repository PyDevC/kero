# Kero-Sine

Kero-Sine: GPU Accelerated SQL Query Engine

KeroSine is MLIR-based SQL Compiler for Deep Learning Libraries such as PyTorch.
It works as a DataLoader that converts the query results in tensors and optimizes whole process to deliver better experience when working with SQL related Deep Learning pipelines.

> Dependencies, API, and docs will be updated at first pre-release 

## How to build

## How to build llvm

For this project you need to build LLVM from source.
Install all Dependencies as per system requirements

```bash
git clone https://github.com/llvm/llvm-project.git

cmake -S llvm -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                                -DLLVM_ENABLE_PROJECTS="clang;lld;mlir;clang-tools-extra;compiler-rt;llvm;lldb;" \
                                -DLLVM_USE_LINKER=lld \
                                -DLLVM_PARALLEL_LINK_JOBS=3 \
                                -DLLVM_PARALLEL_TABLEGEN_JOBS=5
```

### MLIR dialect build

Before working with this project make sure you have mlir build from source.

```bash
export THIRDPARTY_LLVM_DIR"="/path/to/llvm-project"
# To build python library
bash scripts/py_build.sh
```
