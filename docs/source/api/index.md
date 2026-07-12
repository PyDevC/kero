# API Reference

```{toctree}
:maxdepth: 4

kero
```

## Subpackages

| Package | Description |
|---------|-------------|
| {mod}`kero.arrow` | PyArrow data utilities, dataset management, and sample generators |
| {mod}`kero.engine` | SQL parser, MLIR codegen, and JIT execution engine |
| {mod}`kero.tensors` | Typed tensor abstractions for relational data |

## Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| {class}`~kero.arrow.data.Dataset` | `kero.arrow` | PyArrow table container |
| {class}`~kero.engine.parser.Parser` | `kero.engine` | SQL string to internal AST |
| {class}`~kero.engine.codegen.IRGen` | `kero.engine` | MLIR IR generator from DB AST |
| {class}`~kero.engine.execution.KeroEngine` | `kero.engine` | JIT execution engine |
