# Kero-Sine

**MLIR-based GPU Accelerated SQL Query Engine for Deep Learning**

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

> **Note:** Dependencies, full API documentation, and additional docs will be updated at the first pre-release.

---

## Overview

KeroSine is an MLIR-based SQL compiler designed to integrate structured data workflows with deep learning pipelines. It functions as a high-performance DataLoader that converts SQL query results directly into tensors, leveraging GPU acceleration to optimize the entire SQL-to-tensor pipeline for frameworks such as PyTorch.

**Key capabilities:**

- Compiles and executes SQL queries using an MLIR-based backend
- Converts query results directly into GPU-ready tensors
- Optimizes end-to-end SQL and deep learning pipelines
- Designed for seamless integration with PyTorch and similar libraries

---

## Architecture

```
SQL Query
    │
    ▼
┌──────────────────┐
│  KeroSine        │  ← MLIR-based SQL Compiler
│  Compiler        │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  GPU Engine      │  ← GPU Accelerated Execution
└──────────────────┘
    │
    ▼
Tensors (PyTorch Compatible)
```

---

## Project Structure

```
kero/
├── include/        # Public header files
├── kero/           # Core source code
├── lib/            # Library files
├── docs/           # Documentation
├── scripts/        # Build and utility scripts
├── test/           # Test suite
├── tools/          # Developer tools
├── CMakeLists.txt  # CMake build configuration
├── pyproject.toml  # Python package configuration
└── .clang-format   # Code formatting rules
```

---

## Prerequisites

Before building this project, ensure the following are installed and configured on your system:

- CMake
- Ninja build system
- LLVM / MLIR (built from source — see instructions below)
- CUDA Toolkit (for GPU support)
- Python 3.x

---

## Build Instructions

### Step 1 — Build LLVM from Source

This project requires LLVM to be built from source with MLIR support enabled.

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

> **Tip:** LLVM compilation is resource-intensive. If your system has limited RAM, reduce `DLLVM_PARALLEL_LINK_JOBS` to `1` or `2` to avoid out-of-memory errors.

---

### Step 2 — Build Kero-Sine

Once LLVM is built, set the path and run the Python build script:

```bash
export THIRDPARTY_LLVM_DIR="/path/to/llvm-project"

bash scripts/py_build.sh
```

---

## Quick Start

```python
import kero

# Execute a SQL query and receive a tensor directly
tensor = kero.query("SELECT feature1, feature2 FROM dataset WHERE label = 1")

# Use the tensor seamlessly with PyTorch
import torch
dataloader = torch.utils.data.DataLoader(tensor, batch_size=32)
```

> Full API reference will be available with the first official release.

---

## Running Tests

```bash
cd build
ctest --output-on-failure
```

---

## Tech Stack

| Component         | Technology              |
|-------------------|-------------------------|
| Compiler Backend  | MLIR / LLVM             |
| GPU Acceleration  | CUDA                    |
| Primary Languages | C++ (44%), Python (44%) |
| Build System      | CMake + Ninja           |
| Dialect           | MLIR (5%)               |
| Scripting         | Shell (1%)              |

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows the formatting rules defined in `.clang-format`.

---

## Links

- [DeepWiki Documentation](https://deepwiki.com/PyDevC/kero)
- [Issue Tracker](https://github.com/PyDevC/kero/issues)
- [Pull Requests](https://github.com/PyDevC/kero/pulls)

---

*Maintained by [PyDevC](https://github.com/PyDevC)*