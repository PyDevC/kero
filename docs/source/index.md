# Kero-Sine

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PyDevC/kero)
![Status](https://img.shields.io/badge/status-pre--release-orange)

**GPU-Accelerated SQL Query Engine built on MLIR.**

Kero-Sine compiles SQL queries through multiple IR levels using MLIR, executing them on CPU with GPU support planned for the future. It uses [SQLGlot](https://github.com/tobymao/sqlglot) for syntax handling across dialects, and [PyArrow](https://arrow.apache.org/docs/python/) for seamless interaction with existing databases.

---

## Key Features

- **MLIR-based compilation pipeline** -- SQL queries are parsed, lowered through custom DB dialects, and compiled to native code via LLVM.
- **Multi-dialect SQL support** -- handles queries from different SQL dialects through SQLGlot.
- **PyArrow integration** -- read and process data from Arrow tables and Parquet files.
- **Typed tensor abstractions** -- represent relational columns as PyTorch tensors for ML workflows.
- **JIT execution** -- compiled queries are executed via an MLIR execution engine with numpy-based result extraction.

---

## Quick Example

```python
from kero.arrow.samples import all_number_dataset
from kero.engine import Parser, codegen
from kero.engine.execution import KeroEngine

dataset = all_number_dataset(size=10_000)
parser = Parser(dataset)

query = "SELECT * FROM employee WHERE age > 20"
query_ast = parser.parse(query)
irgen = codegen.IRGen("get_aged_employee", query_ast)
irgen.emit_ir()
codegen.db_to_llvm_lowering(irgen.module, irgen.context)

exe = KeroEngine(irgen.module, irgen.context)
exe.configure_outputs([0, 1, 2])
results = exe.execute("get_aged_employee", dataset, ["employee"])
numpy_results = exe.results_to_numpy(results)
for id, array in numpy_results.items():
    print(array)
```

---

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
getting_started
api/index
```
