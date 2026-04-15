"""
kero.data.DataLoader
--------------------
PyTorch-style DataLoader for Kero SQL queries.

Provides compile-once semantics: the SQL query is compiled on the first
call to :meth:`__iter__` and the resulting :class:`~kero.engine.compiler.CompiledQuery`
is cached. Subsequent iterations reuse the cached compiled query.

Design notes
~~~~~~~~~~~~
- Batches are returned as ``pa.Table`` slices.
- The SQL query filters/projections are applied at the MLIR compilation
  level, so each batch receives the same query semantics.
- Thread-safety: compilation is idempotent; the cached CompiledQuery is
  read-only after first construction.

Example
~~~~~~~
::

    import pyarrow as pa
    from kero.data import KeroDataLoader

    table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    loader = KeroDataLoader(table, "SELECT id, name WHERE id > 1", batch_size=2)

    for batch in loader:
        print(batch)   # pa.Table slices
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Union

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover — test stubs only
    pa = None  # type: ignore[assignment]

from kero.engine.compiler import Compiler, CompiledQuery
from kero.engine.parser import Parser


@dataclass
class BatchResult:
    """A single batch result with Arrow table and row mask.

    Attributes
    ----------
    table:
        The input Arrow table (full columns, may be masked by caller).
    mask:
        Boolean mask from the compiled query indicating which rows pass
        the filter. ``None`` if the query has no WHERE clause.
    output_schema:
        Arrow schema of the output (filtered/projected columns).
    """
    table: "pa.Table"
    mask: Optional["pa.Array"] = None
    output_schema: Optional["pa.Schema"] = None


class KeroDataLoader:
    """PyTorch-style DataLoader for Kero SQL queries.

    Parameters
    ----------
    table:
        Arrow table to query. Registered under the name ``_data``
        in the internal schema registry.
    sql:
        SQL query to compile and apply to each batch.
    batch_size:
        Number of rows per batch.  Defaults to 32.
    compile_once:
        If ``True`` (default), the SQL query is compiled on the first
        ``__iter__`` call and the resulting ``CompiledQuery`` is reused
        for all subsequent batches.  Set to ``False`` to force
        recompilation on each epoch (useful for development).
    executor_cls:
        Executor class to use for executing the compiled query.
        Defaults to :class:`~kero.engine.executor.Executor`.
    """

    def __init__(
        self,
        table: "pa.Table",
        sql: str,
        *,
        batch_size: int = 32,
        compile_once: bool = True,
        executor_cls=None,
    ) -> None:
        if pa is None:
            raise RuntimeError(
                "KeroDataLoader requires pyarrow. "
                "Install with: pip install pyarrow"
            )
        if not isinstance(table, pa.Table):
            raise TypeError(
                f"Expected pa.Table, got {type(table).__name__}"
            )

        self._table = table
        self._sql = sql
        self._batch_size = batch_size
        self._compile_once = compile_once

        self._compiler: Optional[Compiler] = None
        self._compiled_query: Optional[CompiledQuery] = None
        self._is_compiled = False

        self._parser: Optional[Parser] = None
        self._executor_cls = executor_cls

    @property
    def table(self) -> "pa.Table":
        """The underlying Arrow table."""
        return self._table

    @property
    def sql(self) -> str:
        """The SQL query string."""
        return self._sql

    @property
    def batch_size(self) -> int:
        """Batch size (rows per iteration)."""
        return self._batch_size

    @property
    def num_batches(self) -> int:
        """Number of batches in one epoch.

        Computed as ``ceil(num_rows / batch_size)``.
        """
        n = len(self._table)
        bs = self._batch_size
        return (n + bs - 1) // bs if n > 0 else 0

    def _ensure_compiled(self) -> CompiledQuery:
        """Compile the SQL query if not already done.

        Implements compile-once semantics: the first call triggers
        compilation and caches the result; subsequent calls return the
        cached value unless ``compile_once=False``.
        """
        if self._compiled_query is not None and self._compile_once:
            return self._compiled_query

        self._compiler = Compiler()
        self._parser = Parser()
        self._parser.attach_module(self._compiler.module)
        self._parser.registry.register("_data", self._table)

        meta = self._parser.parse(self._sql)
        compiled = self._compiler.compile(meta)
        self._compiled_query = compiled
        self._is_compiled = True

        return compiled

    def __iter__(self) -> Iterator[BatchResult]:
        """Yield batches of Arrow data with the query applied.

        On the first call, the SQL query is compiled (compile-once
        semantics) and the resulting CompiledQuery is cached.

        Yields
        ------
        BatchResult
            Each batch contains the Arrow table slice and the output
            schema.
        """
        compiled = self._ensure_compiled()
        n = len(self._table)
        bs = self._batch_size

        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch_table = self._table.slice(start, end - start)

            result = BatchResult(
                table=batch_table,
                output_schema=compiled.output_schema,
            )
            yield result

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return self.num_batches

    def __repr__(self) -> str:
        return (
            f"KeroDataLoader("
            f"table={self._table.schema.names!r}, "
            f"sql={self._sql!r}, "
            f"batch_size={self._batch_size}, "
            f"compile_once={self._compile_once})"
        )


class KeroIterableDataset:
    """Base class for Kero iterable datasets.

    Extend this class to provide custom dataset behavior while retaining
    compile-once query semantics.

    Parameters
    ----------
    sql:
        SQL query to compile and apply.
    compile_once:
        If ``True`` (default), compile the query once on first iteration.
    """

    def __init__(self, sql: str, *, compile_once: bool = True) -> None:
        self._sql = sql
        self._compile_once = compile_once
        self._compiled_query: Optional[CompiledQuery] = None
        self._compiler: Optional[Compiler] = None
        self._parser: Optional[Parser] = None

    @property
    def sql(self) -> str:
        """SQL query string."""
        return self._sql

    def register_tables(self, registry) -> None:
        """Register tables into the schema registry.

        Override this method to register one or more tables needed
        by the query.

        Parameters
        ----------
        registry:
            :class:`~kero.arrow.schema_registry.SchemaRegistry` to
            populate.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.register_tables must be implemented"
        )

    def _ensure_compiled(self) -> CompiledQuery:
        """Compile the SQL query if not already done."""
        if self._compiled_query is not None and self._compile_once:
            return self._compiled_query

        self._compiler = Compiler()
        self._parser = Parser()
        self._parser.attach_module(self._compiler.module)
        self.register_tables(self._parser.registry)

        meta = self._parser.parse(self._sql)
        compiled = self._compiler.compile(meta)
        self._compiled_query = compiled

        return compiled

    def __iter__(self) -> Iterator[BatchResult]:
        """Iterate over dataset batches.

        Override in subclass to provide custom iteration logic.
        Default implementation yields from :meth:`_iter_batches`.
        """
        yield from self._iter_batches()

    def _iter_batches(self) -> Iterator[BatchResult]:
        """Default batch iteration (override in subclass)."""
        raise NotImplementedError(
            f"{type(self).__name__}._iter_batches must be implemented"
        )
