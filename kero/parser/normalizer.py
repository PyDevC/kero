from __future__ import annotations
from typing import Optional
import sqlglot.expressions as exp

from kero.parser.ast_nodes import (
    ColumnRef, Literal, BinaryExpr, Expr,
    ScanNode, FilterNode, ProjectNode, RelNode,
)

try:
    import sqlglot.expressions as sg
except ImportError:
    from kero._compat import sqlglot_shim as sg  # type: ignore[no-redef]


class NormalizationError(ValueError):
    """Raised when a SQL construct cannot be normalised to Internal AST."""
    pass


class Normalizer:
    """Converts a parsed SQLGlot Query into an Internal AST RelNode tree.

    Usage
    -----
    >>> norm = Normalizer()
    >>> rel_tree = norm.normalize(query_ast)
    """

    def normalize(self, query: exp.Select) -> RelNode:
        """Main entry point: convert a Query node to a RelNode tree.

        Builds bottom-up: SCAN → FILTER (if WHERE) → PROJECT (always).
        """
        rel = self._extract_source(query)
        rel = self._extract_filter(query, rel)
        rel = self._extract_projection(query, rel)
        return rel

    def _extract_source(self, query: exp.Select) -> ScanNode:
        tables = list(query.find_all(exp.Table))
        if not tables:
            raise NormalizationError("Query has no FROM clause")
        table_name = tables[0].name
        return ScanNode(table=table_name)

    def _extract_filter(self, query: exp.Select, source: RelNode) -> RelNode:
        where_clause = query.find(exp.Where)
        if where_clause is None:
            return source
        predicate = self._visit_expr(where_clause.this)
        return FilterNode(source=source, predicate=predicate)

    def _extract_projection(self, query: exp.Select, source: RelNode) -> ProjectNode:
        exprs = query.args.get("expressions", [])

        # SELECT *
        if len(exprs) == 1 and isinstance(exprs[0], sg.Star):
            return ProjectNode(source=source, columns=[], is_star=True)

        columns = []
        for expr in exprs:
            if isinstance(expr, sg.Column):
                col_ref = self._normalize_column(expr)
                columns.append(col_ref)
            else:
                raise NotImplementedError(
                    f"SELECT expression type {type(expr).__name__!r} "
                    "is not yet supported in projection"
                )
        return ProjectNode(source=source, columns=columns, is_star=False)

    def _visit_expr(self, node) -> Expr:
        """Recursively convert a SQLGlot expression node to an Expr."""

        # Column reference
        if isinstance(node, sg.Column):
            return self._normalize_column(node)

        # Literal
        if isinstance(node, sg.Literal):
            if node.is_string:
                return Literal(value=str(node.this))
            raw = node.this
            if isinstance(raw, str):
                try:
                    return Literal(value=int(raw))
                except ValueError:
                    return Literal(value=float(raw))
            if isinstance(raw, float):
                return Literal(value=raw)
            return Literal(value=int(raw))

        # --- Binary comparisons ---
        if isinstance(node, sg.GT):
            return BinaryExpr(">", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.LT):
            return BinaryExpr("<", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.GTE):
            return BinaryExpr(">=", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.LTE):
            return BinaryExpr("<=", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.EQ):
            return BinaryExpr("=", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, (sg.NEQ,)) or type(node).__name__ in ("NEQ", "NE", "NEQ"):
            return BinaryExpr("<>", self._visit_expr(node.this),
                              self._visit_expr(node.expression))

        if isinstance(node, sg.And):
            return BinaryExpr("AND", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.Or):
            return BinaryExpr("OR", self._visit_expr(node.this),
                              self._visit_expr(node.expression))

        if isinstance(node, sg.Add):
            return BinaryExpr("+", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.Sub):
            return BinaryExpr("-", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.Mul):
            return BinaryExpr("*", self._visit_expr(node.this),
                              self._visit_expr(node.expression))
        if isinstance(node, sg.Div):
            return BinaryExpr("/", self._visit_expr(node.this),
                              self._visit_expr(node.expression))

        raise NotImplementedError(
            f"Expression node type {type(node).__name__!r} "
            "is not yet supported by the Normalizer"
        )

    def _normalize_column(self, node: sg.Column) -> ColumnRef:
        """Convert a SQLGlot Column node to a ColumnRef."""
        if hasattr(node, "this"):
            col_name = node.this.name if hasattr(node.this, "name") else str(node.this)
        else:
            col_name = node.name

        table_qual = getattr(node, "table", None)
        if table_qual and hasattr(table_qual, "name"):
            table_qual = table_qual.name

        return ColumnRef(
            table=table_qual or "",
            column=col_name,
        )
