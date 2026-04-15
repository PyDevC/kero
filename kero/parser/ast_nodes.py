"""
kero.parser.ast_nodes
----------------------
Internal AST dataclasses for the Kero query pipeline.

These nodes decouple the Normalizer (which translates from SQLGlot AST)
from the rest of the pipeline.  The Binder and Emitter only ever see
these types.

Design rules:
  - Pure data; no methods that perform logic.
  - All fields are typed.
  - resolved_type starts as None after normalization; the Schema Binder
    fills it in before the IR Emitter runs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union

# Lazy import to avoid circular dependency; resolved at runtime.
# We store the Arrow DataType as an opaque object.
DataType = object  # type alias placeholder


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

@dataclass
class ColumnRef:
    """A reference to a column inside a table.

    After normalization: resolved_type is None.
    After schema binding: resolved_type is a pa.DataType.
    """
    table: str
    column: str
    resolved_type: Optional[object] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class Literal:
    """A constant value literal.

    value is int, float, or str.
    resolved_type is set by the binder based on Python type.
    """
    value: Union[int, float, str]
    resolved_type: Optional[object] = field(default=None, compare=False)

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)


@dataclass
class BinaryExpr:
    """A binary expression (comparison or arithmetic).

    op is one of: "=", ">", "<", ">=", "<=", "<>", "AND", "OR",
                  "+", "-", "*", "/"
    resolved_type is set by the binder:
      - comparison ops → pa.bool_()
      - arithmetic ops → left operand type
    """
    op: str
    left: "Expr"
    right: "Expr"
    resolved_type: Optional[object] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


# Union type for expressions
Expr = Union[ColumnRef, Literal, BinaryExpr]


# ---------------------------------------------------------------------------
# Relation nodes
# ---------------------------------------------------------------------------

@dataclass
class ScanNode:
    """Represents reading all rows from a table.

    schema is None after normalization, set by the binder to the
    pa.Schema of the table from the Schema Registry.
    """
    table: str
    schema: Optional[object] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"SCAN({self.table})"


@dataclass
class FilterNode:
    """Represents a WHERE clause filter.

    source is the relation being filtered.
    predicate is the boolean condition expression.
    """
    source: "RelNode"
    predicate: Expr

    def __str__(self) -> str:
        return f"FILTER({self.source}, {self.predicate})"


@dataclass
class ProjectNode:
    """Represents a SELECT list projection.

    columns is the ordered list of column references to emit.
    An empty list means SELECT * (all columns).
    """
    source: "RelNode"
    columns: List[ColumnRef]
    is_star: bool = False

    def __str__(self) -> str:
        if self.is_star:
            return f"PROJECT({self.source}, *)"
        cols = ", ".join(str(c) for c in self.columns)
        return f"PROJECT({self.source}, [{cols}])"


# Union type for relation nodes
RelNode = Union[ScanNode, FilterNode, ProjectNode]
