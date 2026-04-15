"""
kero.arrow.schema_registry
--------------------------
Maps table names to ``pa.Schema`` objects.

The registry is the shared source of truth for column types throughout
the compilation pipeline.  It is owned by the Parser and populated
automatically when the user calls ``KeroEngine.execute()`` with an
Arrow Table (or registers a table explicitly).

Design notes
~~~~~~~~~~~~
- Pure Python, no C++ required — PyArrow gives full schema access.
- ``per_query=True`` mode clears all registered schemas after each
  ``parse()`` call so that column types do not bleed across queries.
- ``register()`` accepts anything with a ``.schema`` attribute
  (``pa.Table``, ``pa.RecordBatch``) as well as a raw ``pa.Schema``.
"""

from __future__ import annotations
from typing import Dict

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover — test stubs only
    pa = None  # type: ignore[assignment]


class SchemaRegistry:
    """``dict[str, pa.Schema]`` with optional per-query reset semantics.

    Parameters
    ----------
    per_query:
        When *True* the registry is cleared after every
        :meth:`on_parse_complete` call (i.e., after each SQL parse).
        This is useful when the same engine instance is reused across
        unrelated queries that share no table names.
    """

    def __init__(self, per_query: bool = False) -> None:
        self._schemas: Dict[str, object] = {}   # str → pa.Schema
        self.per_query = per_query

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, table_name: str, table) -> None:
        """Register *table* under *table_name*.

        Parameters
        ----------
        table_name:
            The name used in SQL (``FROM <table_name>``).
        table:
            A ``pa.Table``, ``pa.RecordBatch``, or any object that
            exposes a ``.schema`` attribute.  Pass a raw ``pa.Schema``
            to register a schema without data.
        """
        if pa is not None and isinstance(table, pa.Schema):
            self._schemas[table_name] = table
        else:
            self._schemas[table_name] = table.schema

    def reset(self) -> None:
        """Remove all registered schemas."""
        self._schemas.clear()

    def on_parse_complete(self) -> None:
        """Hook invoked by the Parser after each :meth:`parse` call.

        Clears the registry when ``per_query`` mode is active.
        """
        if self.per_query:
            self.reset()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, table_name: str):
        """Return the ``pa.Schema`` for *table_name*.

        Raises
        ------
        KeyError
            If *table_name* has not been registered.
        """
        if table_name not in self._schemas:
            raise KeyError(
                f"Table {table_name!r} is not registered in the Schema Registry. "
                f"Available tables: {list(self._schemas.keys())}"
            )
        return self._schemas[table_name]

    def contains(self, table_name: str) -> bool:
        """Return *True* if *table_name* is registered."""
        return table_name in self._schemas

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._schemas)

    def __contains__(self, table_name: str) -> bool:
        return table_name in self._schemas

    def __repr__(self) -> str:
        keys = list(self._schemas.keys())
        return f"SchemaRegistry(tables={keys}, per_query={self.per_query})"
