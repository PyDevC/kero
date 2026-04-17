from dataclasses import dataclass, field
from typing import List, Optional, Union
import pyarrow as pa

@dataclass
class ColumnRef:
    table: str
    column: str
    dtype: Optional[pa.Schema] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class Literal:
    value: Union[int, float, str]
    dtype: Optional[pa.Schema] = field(default=None, compare=False)

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)


@dataclass
class BinaryExpr:
    op: str
    left: "Expr"
    right: "Expr"
    dtype: Optional[pa.Schema] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


Expr = Union[ColumnRef, Literal, BinaryExpr]


@dataclass
class ScanNode:
    table: str
    schema: Optional[pa.Schema] = field(default=None, compare=False)

    def __str__(self) -> str:
        return f"SCAN({self.table})"


@dataclass
class FilterNode:
    source: "RelNode"
    predicate: Expr

    def __str__(self) -> str:
        return f"FILTER({self.source}, {self.predicate})"


@dataclass
class ProjectNode:
    source: "RelNode"
    columns: List[ColumnRef]
    is_star: bool = False

    def __str__(self) -> str:
        if self.is_star:
            return f"PROJECT({self.source}, *)"
        cols = ", ".join(str(c) for c in self.columns)
        return f"PROJECT({self.source}, [{cols}])"


RelNode = Union[ScanNode, FilterNode, ProjectNode]
