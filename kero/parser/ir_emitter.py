from __future__ import annotations
from typing import Dict, List, Optional, Tuple, OrderedDict as ODict
from collections import OrderedDict

from kero.parser.ast_nodes import (
    ColumnRef, Literal, BinaryExpr, Expr,
    ScanNode, FilterNode, ProjectNode, RelNode,
)
from kero._mlir import (
    Value, Module, db_table_type, db_column_type,
    db_result_type, db_row_type,
)


class IRQueryMeta:
    """Metadata produced by the IR Emitter.

    The C++ KeroModule.compile_cpu() will later attach the jit_fn and
    engine_ref to produce the full CompiledQuery.  For the Python-only
    execution path this object is passed directly to the Executor.
    """

    def __init__(
        self,
        ir_text: str,
        table_name: str,
        referenced_columns: List[str],
        output_schema,
        n_cols: int,
    ):
        self.ir_text = ir_text
        self.table_name = table_name
        self.referenced_columns = referenced_columns  # ordered, as declared
        self.output_schema = output_schema
        self.n_cols = n_cols          # number of columns (C dimension)

    def __repr__(self) -> str:
        return (
            f"IRQueryMeta(table={self.table_name!r}, "
            f"columns={self.referenced_columns}, "
            f"n_cols={self.n_cols})"
        )


class IREmitter:
    def __init__(self):
        self._arg_map: Dict[str, Value] = {}
        self._ssa_cnt = 0
        self._lines: List[str] = []   # function body lines
        self._func_args: List[Value] = []

    def emit(self, root: RelNode, module: Module) -> IRQueryMeta:
        """Emit a func.func for *root* into *module*.

        Returns the IRQueryMeta describing argument order and IR text.
        """
        self._arg_map = {}
        self._ssa_cnt = 0
        self._lines = []
        self._func_args = []

        scan_node = self._find_scan(root)
        table_name = scan_node.table
        schema = scan_node.schema

        seen_cols: ODict[str, ColumnRef] = OrderedDict()
        self._collect_columns(root, seen_cols)

        # arg0 = table
        table_type = db_table_type(table_name)
        table_val = Value("%arg0", table_type)
        self._func_args.append(table_val)
        self._arg_map[f"__table__{table_name}"] = table_val

        # arg1..N = columns in encounter order
        for i, (col_key, col_ref) in enumerate(seen_cols.items(), start=1):
            mlir_elem = self._arrow_to_mlir(col_ref.resolved_type)
            col_type = db_column_type(col_ref.table, col_ref.column, mlir_elem)
            col_val = Value(f"%arg_col{i}", col_type)
            self._func_args.append(col_val)
            self._arg_map[col_key] = col_val

        result_val = self._emit_rel(root)
        self._lines.append(f"return {result_val} : !db.result")

        arg_strs = [f"    {v} : {v.mlir_type}" for v in self._func_args]
        func_lines = [
            f"func.func @query(",
            ",\n".join(arg_strs),
            f") -> !db.result {{",
        ] + [f"    {ln}" for ln in self._lines] + ["}"]
        func_text = "\n".join(func_lines)

        module.add_function(func_text)
        referenced_columns = [ref.column for ref in seen_cols.values()]
        output_schema = schema

        return IRQueryMeta(
            ir_text=func_text,
            table_name=table_name,
            referenced_columns=referenced_columns,
            output_schema=output_schema,
            n_cols=len(schema) if schema else 0,
        )

    def _emit_rel(self, node: RelNode) -> Value:
        if isinstance(node, ScanNode):
            return self._emit_scan(node)
        if isinstance(node, FilterNode):
            return self._emit_filter(node)
        if isinstance(node, ProjectNode):
            return self._emit_project(node)
        raise NotImplementedError(
            f"IREmitter._emit_rel: unknown node type {type(node).__name__!r}"
        )

    def _emit_scan(self, node: ScanNode) -> Value:
        table_val = self._arg_map[f"__table__{node.table}"]
        result_name = self._fresh("%scan")
        result_val = Value(result_name, "!db.result")
        self._lines.append(
            f"{result_name} = db.scan {table_val} "
            f": {db_table_type(node.table)} -> !db.result"
        )
        return result_val

    def _emit_filter(self, node: FilterNode) -> Value:
        source_val = self._emit_rel(node.source)
        result_name = self._fresh("%filtered")

        row_val = Value("%row", "!db.row")
        region_lines: List[str] = []

        pred_val = self._emit_expr_in_region(
            node.predicate, row_val, region_lines
        )
        region_lines.append(f"db.return {pred_val} : i1")

        region_text = "\n".join(f"        {ln}" for ln in region_lines)

        filter_lines = [
            f"{result_name} = db.filter {source_val} {{",
            f"    ^bb0({row_val} : !db.row):",
            region_text,
            f"}} : (!db.result) -> !db.result",
        ]
        self._lines += filter_lines
        return Value(result_name, "!db.result")

    def _emit_project(self, node: ProjectNode) -> Value:
        source_val = self._emit_rel(node.source)
        if node.is_star or not node.columns:
            return source_val

        col_args = " ".join(
            str(self._arg_map[self._col_key(c)]) for c in node.columns
        )
        result_name = self._fresh("%projected")
        self._lines.append(
            f"{result_name} = db.project {source_val}"
            f" : !db.result -> !db.result"
        )
        return Value(result_name, "!db.result")

    def _emit_expr_in_region(
        self, node: Expr, row_val: Value, out: List[str]
    ) -> Value:
        """Emit ops for *node* appended to *out*, return the result Value."""

        if isinstance(node, ColumnRef):
            col_arg = self._arg_map[self._col_key(node)]
            mlir_elem = self._arrow_to_mlir(node.resolved_type)
            res = self._fresh("%col_val")
            out.append(
                f"{res} = db.getcol {row_val}, {col_arg} "
                f": (!db.row, {col_arg.mlir_type}) -> {mlir_elem}"
            )
            return Value(res, mlir_elem)

        if isinstance(node, Literal):
            mlir_elem = self._arrow_to_mlir(node.resolved_type)
            res = self._fresh("%lit")
            if isinstance(node.value, float):
                out.append(
                    f"{res} = arith.constant {node.value:.6e} : {mlir_elem}"
                )
            else:
                out.append(
                    f"{res} = arith.constant {node.value} : {mlir_elem}"
                )
            return Value(res, mlir_elem)

        if isinstance(node, BinaryExpr):
            left_val = self._emit_expr_in_region(node.left, row_val, out)
            right_val = self._emit_expr_in_region(node.right, row_val, out)
            return self._emit_binary_op(node.op, left_val, right_val, out)

        raise NotImplementedError(
            f"IREmitter._emit_expr_in_region: {type(node).__name__!r}"
        )

    def _emit_binary_op(
        self, op: str, left: Value, right: Value, out: List[str]
    ) -> Value:
        res = self._fresh("%cmp")
        is_float = "f" in left.mlir_type or "f64" in left.mlir_type

        op_map_int = {
            ">":  "arith.cmpi sgt",
            "<":  "arith.cmpi slt",
            ">=": "arith.cmpi sge",
            "<=": "arith.cmpi sle",
            "=":  "arith.cmpi eq",
            "<>": "arith.cmpi ne",
        }
        op_map_float = {
            ">":  "arith.cmpf ogt",
            "<":  "arith.cmpf olt",
            ">=": "arith.cmpf oge",
            "<=": "arith.cmpf ole",
            "=":  "arith.cmpf oeq",
            "<>": "arith.cmpf one",
        }
        arith_map = {
            "+":   "arith.addi" if not is_float else "arith.addf",
            "-":   "arith.subi" if not is_float else "arith.subf",
            "*":   "arith.muli" if not is_float else "arith.mulf",
            "/":   "arith.divsi" if not is_float else "arith.divf",
            "AND": "arith.andi",
            "OR":  "arith.ori",
        }

        if op in op_map_float and is_float:
            instr = op_map_float[op]
            out.append(
                f"{res} = {instr}, {left}, {right} : {left.mlir_type}"
            )
            return Value(res, "i1")
        elif op in op_map_int:
            instr = op_map_int[op]
            out.append(
                f"{res} = {instr}, {left}, {right} : {left.mlir_type}"
            )
            return Value(res, "i1")
        elif op in arith_map:
            instr = arith_map[op]
            out.append(
                f"{res} = {instr} {left}, {right} : {left.mlir_type}"
            )
            return Value(res, left.mlir_type)
        else:
            raise NotImplementedError(
                f"IREmitter: binary op {op!r} not yet implemented"
            )

    def _fresh(self, prefix: str) -> str:
        name = f"{prefix}{self._ssa_cnt}"
        self._ssa_cnt += 1
        return name

    @staticmethod
    def _col_key(ref: ColumnRef) -> str:
        return f"{ref.table}.{ref.column}"

    def _collect_columns(
        self, node: RelNode, seen: ODict
    ) -> None:
        """Depth-first walk; collect ColumnRefs in encounter order."""
        if isinstance(node, ScanNode):
            pass  # no expressions
        elif isinstance(node, FilterNode):
            self._collect_columns(node.source, seen)
            self._collect_columns_expr(node.predicate, seen)
        elif isinstance(node, ProjectNode):
            self._collect_columns(node.source, seen)
            for col in node.columns:
                key = self._col_key(col)
                if key not in seen:
                    seen[key] = col

    def _collect_columns_expr(self, expr: Expr, seen: ODict) -> None:
        if isinstance(expr, ColumnRef):
            key = self._col_key(expr)
            if key not in seen:
                seen[key] = expr
        elif isinstance(expr, BinaryExpr):
            self._collect_columns_expr(expr.left, seen)
            self._collect_columns_expr(expr.right, seen)

    @staticmethod
    def _arrow_to_mlir(dtype) -> str:
        """Convert an Arrow DataType to its MLIR equivalent string."""
        if dtype is None:
            return "i64"
        name = str(dtype)
        _map = {
            "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
            "uint8": "i8", "uint16": "i16", "uint32": "i32", "uint64": "i64",
            "float32": "f32", "float64": "f64",
            "bool_": "i1", "bool": "i1",
            "string": "i64",   # strings as hash
        }
        return _map.get(name, "i64")

    def _find_scan(self, node: RelNode) -> ScanNode:
        if isinstance(node, ScanNode):
            return node
        if hasattr(node, "source"):
            return self._find_scan(node.source)
        raise ValueError("No ScanNode found in relation tree")
