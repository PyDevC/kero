"""demo_pipeline.py — End-to-end pipeline demo for Kero SQL query engine.

Demonstrates:
  1. SchemaRegistry — register Arrow tables and inspect schemas
  2. Parser + Compiler — parse SQL → emit db-dialect IR → compile to JIT
  3. IR diagnostics — verify_ir() and lower_to_llvm_ir()
  4. KeroDataLoader — PyTorch-style batched iteration over Arrow data
  5. Multi-table patterns — separate Compiler instances per table
  6. Error handling — what happens when the C++ extension is absent

Run without the C++ extension built (pure Python fallback):
    python demo_pipeline.py

Run with the C++ extension built:
    PYTHONPATH=build/lib python demo_pipeline.py
"""

from __future__ import annotations

import textwrap
import traceback

import numpy as np
import pyarrow as pa

# ── kero imports ────────────────────────────────────────────────────────────
from kero.arrow.schema_registry import SchemaRegistry
from kero.engine.parser import Parser
from kero.engine.compiler import Compiler, _HAVE_KERO
from kero.data import KeroDataLoader

# ── helpers ─────────────────────────────────────────────────────────────────

BANNER = "=" * 60


def section(title: str) -> None:
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def _make_employees_table(n: int = 20) -> pa.Table:
    """Small synthetic employees table."""
    rng = np.random.default_rng(0)
    return pa.table({
        "id":         pa.array(range(1, n + 1),                type=pa.int64()),
        "age":        pa.array(rng.integers(22, 65, n).tolist(), type=pa.int64()),
        "salary":     pa.array(rng.integers(30_000, 150_000, n).tolist(), type=pa.float64()),
        "department": pa.array(rng.choice(["eng", "sales", "hr", "ops"], n).tolist()),
        "active":     pa.array(rng.choice([True, False], n).tolist()),
    })


def _make_orders_table(n: int = 40) -> pa.Table:
    """Small synthetic orders table."""
    rng = np.random.default_rng(1)
    return pa.table({
        "order_id":   pa.array(range(1001, 1001 + n), type=pa.int64()),
        "emp_id":     pa.array(rng.integers(1, 21, n).tolist(), type=pa.int64()),
        "amount":     pa.array(rng.uniform(10.0, 9_999.0, n).round(2).tolist(), type=pa.float64()),
        "paid":       pa.array(rng.choice([True, False], n).tolist()),
    })


# ── demo 1: SchemaRegistry ──────────────────────────────────────────────────

def demo_schema_registry() -> None:
    section("Demo 1 — SchemaRegistry")

    employees = _make_employees_table()
    orders = _make_orders_table()

    registry = SchemaRegistry()
    registry.register("employees", employees)
    registry.register("orders", orders)

    print("Registered tables:", registry.tables() if hasattr(registry, "tables") else
          list(registry._schemas.keys()))

    schema = registry.get("employees")
    print("\nemployees schema:")
    for field in schema:
        print(f"  {field.name:15s} {field.type}")

    # per_query=True clears after each parse — demo only
    per_q = SchemaRegistry(per_query=True)
    per_q.register("tmp", employees)
    per_q.on_parse_complete()
    print("\nAfter on_parse_complete (per_query=True), tables:", end=" ")
    try:
        print(list(per_q._schemas.keys()))
    except AttributeError:
        print("<cleared>")


# ── demo 2: Parser → IRQueryMeta (pure Python) ─────────────────────────────

def demo_parser() -> None:
    section("Demo 2 — Parser: SQL → IRQueryMeta")

    employees = _make_employees_table()

    compiler = Compiler() if _HAVE_KERO else None

    # Build parser and wire up the shared module
    parser = Parser()
    if compiler:
        parser.attach_module(compiler.module)

    parser.registry.register("employees", employees)

    queries = [
        "SELECT id, age, salary FROM _data WHERE age > 30",
    ]

    for sql in queries:
        print(f"\nSQL : {sql}")
        if compiler is None:
            print("  (skipping compilation — _kero extension not built)")
            continue
        try:
            meta = parser.parse(sql)
            print(f"  table_name         : {meta.table_name}")
            print(f"  referenced_columns : {meta.referenced_columns}")
            print(f"  n_cols             : {meta.n_cols}")
            print(f"  output_schema      : {meta.output_schema}")
            compiler.reset_module()
        except Exception as exc:
            print(f"  ERROR: {exc}")


# ── demo 3: Compiler diagnostics ────────────────────────────────────────────

def demo_compiler_diagnostics() -> None:
    section("Demo 3 — Compiler Diagnostics (verify_ir / lower_to_llvm_ir)")

    if not _HAVE_KERO:
        print("Skipping — _kero extension not built.")
        return

    employees = _make_employees_table()

    compiler = Compiler()
    parser = Parser()
    parser.attach_module(compiler.module)
    parser.registry.register("employees", employees)

    sql = "SELECT id, salary FROM employees WHERE salary > 50000"
    meta = parser.parse(sql)

    # verify_ir
    ir_text = compiler.module.get_ir()
    err = compiler.verify_ir(ir_text)
    if err:
        print(f"verify_ir ERROR: {err}")
    else:
        print("verify_ir: OK")

    # Print first few lines of the db-dialect IR
    lines = ir_text.strip().splitlines()
    preview = "\n".join(lines[:min(20, len(lines))])
    print(f"\ndb-dialect IR (first 20 lines):\n{textwrap.indent(preview, '  ')}")
    if len(lines) > 20:
        print(f"  ... ({len(lines) - 20} more lines)")

    # lower_to_llvm_ir
    try:
        llvm_ir = compiler.lower_to_llvm_ir(ir_text)
        llvm_lines = llvm_ir.strip().splitlines()
        preview = "\n".join(llvm_lines[:min(10, len(llvm_lines))])
        print(f"\nLLVM dialect IR (first 10 lines):\n{textwrap.indent(preview, '  ')}")
    except RuntimeError as exc:
        print(f"lower_to_llvm_ir ERROR: {exc}")


# ── demo 4: compile_cpu + call_jit ──────────────────────────────────────────

def demo_compile_and_execute() -> None:
    section("Demo 4 — compile_cpu → call_jit")

    if not _HAVE_KERO:
        print("Skipping — _kero extension not built.")
        return

    employees = _make_employees_table()

    compiler = Compiler()
    parser = Parser()
    parser.attach_module(compiler.module)
    parser.registry.register("employees", employees)

    sql = "SELECT id, age, salary FROM employees WHERE age > 35"
    meta = parser.parse(sql)
    compiled = compiler.compile(meta)

    print(f"CompiledQuery : {compiled}")
    print(f"table_name    : {compiled.table_name}")
    print(f"n_cols        : {compiled.n_cols}")
    print(f"columns       : {compiled.referenced_columns}")

    # The Executor normally handles the void** ABI details.  Here we
    # demonstrate that the CompiledQuery object is properly formed.
    print("\ncompiledQuery.jit_fn is callable:", callable(compiled.jit_fn.call_jit))


# ── demo 5: KeroDataLoader ───────────────────────────────────────────────────

def demo_data_loader() -> None:
    section("Demo 5 — KeroDataLoader (batched iteration)")

    employees = _make_employees_table(n=100)

    loader = KeroDataLoader(
        employees,
        "SELECT id, age, salary FROM _data WHERE salary > 60000",
        batch_size=16,
        compile_once=True,
    )

    print(f"Table rows   : {len(employees)}")
    print(f"Batch size   : {loader.batch_size}")
    print(f"Num batches  : {loader.num_batches}")
    print(f"Loader repr  : {loader}")

    try:
        for i, batch in enumerate(loader):
            print(f"  Batch {i:02d}: {len(batch.table)} rows, "
                  f"schema={batch.table.schema.names}")
    except RuntimeError as exc:
        # Expected when _kero extension is not built — compilation step
        print(f"  (compile step skipped — {exc})")
        # Still show that the batching logic itself works by iterating
        # without the filter applied:
        n = len(employees)
        bs = loader.batch_size
        print(f"  Raw batching demo ({n} rows, batch={bs}):")
        for start in range(0, n, bs):
            end = min(start + bs, n)
            print(f"    rows [{start}:{end}]")


# ── demo 6: Multi-table pattern ──────────────────────────────────────────────

def demo_multi_table() -> None:
    section("Demo 6 — Multi-table pattern (one Compiler per table)")

    employees = _make_employees_table()
    orders = _make_orders_table()

    # Each table gets its own Compiler (→ its own MLIRContext).
    # This is the recommended pattern when queries over different tables
    # need to coexist in the same Python process.
    configs = [
        ("employees", employees, "SELECT id, salary FROM employees WHERE salary > 90000"),
        ("orders",    orders,    "SELECT order_id, amount FROM orders WHERE amount > 5000"),
    ]

    for table_name, table, sql in configs:
        print(f"\nTable  : {table_name}")
        print(f"SQL    : {sql}")
        if not _HAVE_KERO:
            print("  (skipping compilation — _kero extension not built)")
            continue
        try:
            compiler = Compiler()
            parser = Parser()
            parser.attach_module(compiler.module)
            parser.registry.register(table_name, table)
            meta = parser.parse(sql)
            compiled = compiler.compile(meta)
            print(f"  compiled : {compiled}")
        except Exception as exc:
            print(f"  ERROR: {exc}")


# ── demo 7: GPU compile_gpu stub ─────────────────────────────────────────────

def demo_compile_gpu() -> None:
    section("Demo 7 — compile_gpu (NVIDIA/AMD targets)")

    if not _HAVE_KERO:
        print("Skipping — _kero extension not built.")
        return

    employees = _make_employees_table()

    for target in ("nvidia", "amd"):
        print(f"\nTarget: {target}")
        try:
            compiler = Compiler()
            parser = Parser()
            parser.attach_module(compiler.module)
            parser.registry.register("employees", employees)

            sql = "SELECT id, salary FROM employees WHERE salary > 70000"
            meta = parser.parse(sql)
            ir_text = compiler.module.get_ir()

            # Call compile_gpu directly on the underlying KeroModule.
            # The Compiler.compile() always calls compile_cpu; for GPU you
            # call the KeroModule binding directly.
            from kero import _kero
            km = _kero.KeroModule()
            compiled = km.compile_gpu(
                ir_text,
                meta.table_name,
                meta.referenced_columns,
                meta.n_cols,
                target,
            )
            print(f"  compiled : {compiled}")
        except RuntimeError as exc:
            # Expected unless a GPU runtime (CUDA/ROCm) is present.
            print(f"  RuntimeError (expected without GPU runtime): {exc}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nKero SQL Query Engine — Pipeline Demo")
    print(f"_kero C++ extension available: {_HAVE_KERO}")

    demo_schema_registry()
    demo_parser()
    demo_compiler_diagnostics()
    demo_compile_and_execute()
    demo_data_loader()
    demo_multi_table()
    demo_compile_gpu()

    print(f"\n{BANNER}")
    print("  All demos complete.")
    print(BANNER)


if __name__ == "__main__":
    main()
