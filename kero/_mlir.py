"""
kero._mlir
----------
Pure-Python stubs for the MLIR type system and Module used by the
IR Emitter.  These objects carry enough information for the emitter to
build textual MLIR IR; the real MLIR C++ context is created later in
KeroModule.compile_cpu() which ingests the IR text.

Classes
-------
Value
    Represents a single SSA value in the IR (a name + an MLIR type
    string).  Renders as its name when used in f-strings.

Module
    Accumulates func.func definitions and can render the complete
    module as a textual MLIR string.

Helper functions
----------------
db_table_type(name)         → "!db.table<\"name\">"
db_column_type(t, c, elem)  → "!db.column<\"t\", \"c\", elem>"
db_result_type()            → "!db.result"
db_row_type()               → "!db.row"
"""

from __future__ import annotations
from typing import List


# ---------------------------------------------------------------------------
# Value — a single SSA name + its MLIR type
# ---------------------------------------------------------------------------

class Value:
    """An MLIR SSA value.

    Parameters
    ----------
    name:
        The SSA name, e.g. ``%arg0`` or ``%scan0``.
    mlir_type:
        The MLIR type string, e.g. ``"!db.table<\\"user\\">"`` or
        ``"i32"``.
    """

    __slots__ = ("name", "mlir_type")

    def __init__(self, name: str, mlir_type: str) -> None:
        self.name = name
        self.mlir_type = mlir_type

    def __str__(self) -> str:          # used in IR f-strings
        return self.name

    def __repr__(self) -> str:
        return f"Value({self.name!r}, {self.mlir_type!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self.name == other.name and self.mlir_type == other.mlir_type

    def __hash__(self) -> int:
        return hash((self.name, self.mlir_type))


# ---------------------------------------------------------------------------
# Module — accumulates func.func bodies and renders the IR text
# ---------------------------------------------------------------------------

class Module:
    """Lightweight representation of an ``mlir::ModuleOp``.

    The IR Emitter calls :meth:`add_function` to register each
    compiled query function.  :meth:`get_ir` returns the complete
    textual module that KeroModule.compile_cpu() will parse.

    The module automatically registers the ``db`` dialect header so
    that ``mlir-opt`` / ``kero-opt`` can round-trip it.
    """

    def __init__(self) -> None:
        self._functions: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_function(self, func_text: str) -> None:
        """Append a ``func.func`` definition to this module.

        Parameters
        ----------
        func_text:
            Complete textual MLIR for one ``func.func`` block,
            including the closing ``}``.
        """
        self._functions.append(func_text)

    def get_ir(self) -> str:
        """Return the complete module as a textual MLIR string.

        The returned string can be fed directly to
        ``mlir::parseSourceString`` or ``KeroModule.compile_cpu()``.
        """
        body = "\n\n".join(self._functions)
        return f"module {{\n{body}\n}}"

    def clear(self) -> None:
        """Remove all accumulated functions (useful between queries)."""
        self._functions.clear()

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        return f"Module(functions={len(self._functions)})"


# ---------------------------------------------------------------------------
# db-dialect type constructors
# ---------------------------------------------------------------------------

def db_table_type(table_name: str) -> str:
    """Return the MLIR type string for ``!db.table<"name">``."""
    return f'!db.table<"{table_name}">'


def db_column_type(table_name: str, col_name: str, elem_type: str) -> str:
    """Return the MLIR type string for ``!db.column<"t", "c", elem>``."""
    return f'!db.column<"{table_name}", "{col_name}", {elem_type}>'


def db_result_type() -> str:
    """Return the MLIR type string ``!db.result``."""
    return "!db.result"


def db_row_type() -> str:
    """Return the MLIR type string ``!db.row``."""
    return "!db.row"
