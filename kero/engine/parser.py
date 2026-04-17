from __future__ import annotations
from typing import Optional

from kero.arrow.schema_registry import SchemaRegistry
from kero.parser.normalizer import Normalizer
from kero.parser.schema_binder import SchemaBinder
from kero.parser.ir_emitter import IREmitter, IRQueryMeta
from kero._mlir import Module

try:
    import sqlglot
    def _parse_one(sql: str):
        return sqlglot.parse_one(sql)
except ImportError:
    from kero._compat.sqlglot_shim import parse_one as _parse_one  # type: ignore


class Parser:
    def __init__(self, per_query: bool = False):
        self.registry = SchemaRegistry(per_query=per_query)
        self._normalizer = Normalizer()
        self._module: Optional[Module] = None   # injected by Compiler

    def attach_module(self, module: Module) -> None:
        """Inject the shared MLIR Module (created by the Compiler)."""
        self._module = module

    def parse(self, sql: str) -> IRQueryMeta:
        """Parse sql and emit db-dialect IR into the shared module.

        Returns
        -------
        IRQueryMeta
            Metadata describing the emitted function: argument order,
            table name, output schema, and the IR text itself.
        """
        if self._module is None:
            raise RuntimeError(
                "Parser has no attached MLIR module. "
                "Call parser.attach_module(module) before parse()."
            )

        query_ast = _parse_one(sql)
        internal_ast = self._normalizer.normalize(query_ast)
        binder = SchemaBinder(self.registry)
        typed_ast = binder.bind(internal_ast)
        emitter = IREmitter()
        meta = emitter.emit(typed_ast, self._module)
        self.registry.on_parse_complete()

        return meta
