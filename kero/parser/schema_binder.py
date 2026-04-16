# Type resolution

import pyarrow as pa

from typing import Dict

from .ast_nodes import *
from ..arrow.schema_registry import SchemaRegistry


class SchemaBindingError(ValueError):
    """Raised when a SQL column or table reference cannot be resolved."""
    pass


class SchemaBinder:
    def __init__(self, registry: SchemaRegistry) -> None:
        self._registry = registry
        self._col_types: Dict[str, Dict[str, object]] = {}

    def bind(self, root: RelNode) -> RelNode:
        # Why would you do this?
        self._col_types = {}
        return self._bind_rel(root)

    def _bind_rel(self, node: RelNode) -> RelNode:
        if isinstance(node, ScanNode):
            return self._bind_scan(node)
        elif isinstance(node, FilterNode):
            return self._bind_filter(node)
        elif isinstance(node, ProjectNode):
            return self._bind_project(node)

        raise NotImplementedError(
            f"SchemaBinder._bind_rel: unknown node {type(node).__name__!r}"
        )

    def _bind_scan(self, node: ScanNode) -> ScanNode:
        """Attach schema to ScanNode and populate the column-type cache."""
        table_name = node.table
        if not self._registry.registed(table_name):
            raise SchemaBindingError(
                f"Table {table_name!r} is not registered. "
                f"Register it via the Schema Registry before parsing."
            )
        schema = self._registry.get(table_name)
        node.schema = schema

        # Build col_name: DataType map
        self._col_types[table_name] = {f.name: f.type for f in schema}
        return node

    def _bind_filter(self, node: FilterNode) -> FilterNode:
        node.source = self._bind_rel(node.source)
        table_name = self._infer_table(node.source)
        node.predicate = self._bind_expr(node.predicate, table_name)
        return node

    def _bind_project(self, node: ProjectNode) -> ProjectNode:
        node.source = self._bind_rel(node.source)
        table_name = self._infer_table(node.source)
        if not node.is_star:
            node.columns = [self._resolve_column(c, table_name) for c in node.columns]
        return node

    def _bind_expr(self, expr: Expr, table_name: str) -> Expr:
        if isinstance(expr, ColumnRef):
            return self._resolve_column(expr, table_name)
        if isinstance(expr, Literal):
            return self._bind_literal(expr)
        if isinstance(expr, BinaryExpr):
            expr.left = self._bind_expr(expr.left, table_name)
            expr.right = self._bind_expr(expr.right, table_name)
            expr.dtype = self._infer_binary_type(expr)
            return expr

        raise NotImplementedError(
            f"SchemaBinder._bind_expr: unknown type {type(expr).__name__!r}"
        )

    def _resolve_column(self, ref: ColumnRef, default_table: str) -> ColumnRef:
        """Look up ref in the schema cache and fill dtype."""
        table_name = ref.table if ref.table else default_table

        if table_name not in self._col_types:
            raise SchemaBindingError(
                f"Unknown table {table_name!r} in column {ref.column!r}."
            )

        col_map = self._col_types[table_name]
        if ref.column not in col_map:
            available = sorted(col_map.keys())
            raise SchemaBindingError(
                f"Column {ref.column!r} not found in table {table_name!r}. "
                f"Available columns: {available}"
            )

        ref.table = table_name
        ref.dtype = col_map[ref.column]
        return ref

    @staticmethod
    def _bind_literal(literal: Literal) -> Literal:
        if isinstance(literal.value, bool):
            literal.dtype = pa.bool_()
        elif isinstance(literal.value, int):
            literal.dtype = pa.int64()
        elif isinstance(literal.value, float):
            literal.dtype = pa.float64()
        else:
            literal.dtype = pa.string()
        return literal

    @staticmethod
    def _infer_binary_type(expr: BinaryExpr):
        _comparison_ops = {">", "<", ">=", "<=", "=", "<>", "AND", "OR"}
        if expr.op in _comparison_ops:
            return pa.bool_() if pa is not None else "bool"
        return expr.left.dtype   # arithmetic: left type propagates

    @staticmethod
    def _infer_table(node: RelNode) -> str:
        if isinstance(node, ScanNode):
            return node.table
        if hasattr(node, "source"):
            return SchemaBinder._infer_table(node.source)
        raise SchemaBindingError("Cannot infer table name from relation tree")
