# Kero-Sine

Kero-Sine: GPU Accelerated SQL Query Engine

KeroSine is MLIR-based SQL Compiler for Deep Learning Libraries such as PyTorch.
It works as a DataLoader that converts the query results in tensors and optimizes whole process to deliver better experience when working with SQL related Deep Learning pipelines.

> Under Development
> Dependencies, API, and docs will be updated at first pre-release 

## How to build

### MLIR dialect build

Before working with this project make sure you have mlir build from source.

```bash
export THIRDPARTY_LLVM_DIR"="/path/to/llvm-project"
bash test/build.sh
```
