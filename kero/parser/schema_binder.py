"""
kero.parser.schema_binder
--------------------------
Stage 2 of the Kero parser pipeline.

Receives the untyped Internal AST produced by the Normalizer and
resolves every ``ColumnRef.resolved_type`` and ``ScanNode.schema``
using the Schema Registry.  After this pass, every node that carries
a type field is guaranteed to be non-``None``.

Classes
-------
SchemaRegistry
    Re-exported here for backward compatibility.

SchemaBinder
    The stage-2 pass.  Constructed with a ``SchemaRegistry`` and
    exposes a single :meth:`bind` method.

SchemaBindingError
    Raised when a column or table reference cannot be resolved.
"""

from __future__ import annotations
from typing import Dict

try:
    import pyarrow as pa
except ImportError:
    pa = None  # type: ignore[assignment]

from kero.parser.ast_nodes import (
    ColumnRef, Literal, BinaryExpr, Expr,
    ScanNode, FilterNode, ProjectNode, RelNode,
)

# Re-export SchemaRegistry so existing code that imports from this
# module continues to work.
from kero.arrow.schema_registry import SchemaRegistry  # noqa: F401


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SchemaBindingError(ValueError):
    """Raised when a SQL column or table reference cannot be resolved."""
    pass


# ---------------------------------------------------------------------------
# SchemaBinder
# ---------------------------------------------------------------------------

class SchemaBinder:
    """Resolves all column types in an Internal AST.

    Parameters
    ----------
    registry:
        A populated ``SchemaRegistry``.  The table referenced in the
        ``ScanNode`` must be registered before :meth:`bind` is called.

    Usage
    -----
    >>> binder = SchemaBinder(registry)
    >>> typed_ast = binder.bind(internal_ast)
    """

    def __init__(self, registry: SchemaRegistry) -> None:
        self._registry = registry
        # table_name -> {col_name -> pa.DataType}  (built during bind)
        self._col_types: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bind(self, root: RelNode) -> RelNode:
        """Walk *root* and fill in all ``resolved_type`` / ``schema`` fields.

        Mutates nodes in place.  Returns the same root.

        Raises
        ------
        SchemaBindingError
            If a referenced table or column is absent from the registry.
        """
        self._col_types = {}
        return self._bind_rel(root)

    # ------------------------------------------------------------------
    # Relation binders
    # ------------------------------------------------------------------

    def _bind_rel(self, node: RelNode) -> RelNode:
        if isinstance(node, ScanNode):
            return self._bind_scan(node)
        if isinstance(node, FilterNode):
            return self._bind_filter(node)
        if isinstance(node, ProjectNode):
            return self._bind_project(node)
        raise NotImplementedError(
            f"SchemaBinder._bind_rel: unknown node {type(node).__name__!r}"
        )

    def _bind_scan(self, node: ScanNode) -> ScanNode:
        """Attach schema to ScanNode and populate the column-type cache."""
        table_name = node.table
        if not self._registry.contains(table_name):
            raise SchemaBindingError(
                f"Table {table_name!r} is not registered. "
                f"Register it via the Schema Registry before parsing."
            )
        schema = self._registry.get(table_name)
        node.schema = schema

        # Build col_name -> DataType map
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
            node.columns = [
                self._resolve_column(c, table_name) for c in node.columns
            ]
        return node

    # ------------------------------------------------------------------
    # Expression binder
    # ------------------------------------------------------------------

    def _bind_expr(self, expr: Expr, table_name: str) -> Expr:
        if isinstance(expr, ColumnRef):
            return self._resolve_column(expr, table_name)
        if isinstance(expr, Literal):
            return self._bind_literal(expr)
        if isinstance(expr, BinaryExpr):
            expr.left = self._bind_expr(expr.left, table_name)
            expr.right = self._bind_expr(expr.right, table_name)
            expr.resolved_type = self._infer_binary_type(expr)
            return expr
        raise NotImplementedError(
            f"SchemaBinder._bind_expr: unknown type {type(expr).__name__!r}"
        )

    def _resolve_column(self, ref: ColumnRef, default_table: str) -> ColumnRef:
        """Look up *ref* in the schema cache and fill ``resolved_type``."""
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
        ref.resolved_type = col_map[ref.column]
        return ref

    @staticmethod
    def _bind_literal(lit: Literal) -> Literal:
        if pa is None:
            # No pyarrow: use sentinel strings _arrow_to_mlir understands
            if isinstance(lit.value, bool):
                lit.resolved_type = "bool"
            elif isinstance(lit.value, float):
                lit.resolved_type = "float64"
            elif isinstance(lit.value, int):
                lit.resolved_type = "int64"
            else:
                lit.resolved_type = "string"
            return lit

        if isinstance(lit.value, bool):
            lit.resolved_type = pa.bool_()
        elif isinstance(lit.value, int):
            lit.resolved_type = pa.int64()
        elif isinstance(lit.value, float):
            lit.resolved_type = pa.float64()
        else:
            lit.resolved_type = pa.string()
        return lit

    @staticmethod
    def _infer_binary_type(expr: BinaryExpr):
        _comparison_ops = {">", "<", ">=", "<=", "=", "<>", "AND", "OR"}
        if expr.op in _comparison_ops:
            return pa.bool_() if pa is not None else "bool"
        return expr.left.resolved_type   # arithmetic: left type propagates

    @staticmethod
    def _infer_table(node: RelNode) -> str:
        if isinstance(node, ScanNode):
            return node.table
        if hasattr(node, "source"):
            return SchemaBinder._infer_table(node.source)
        raise SchemaBindingError("Cannot infer table name from relation tree")
