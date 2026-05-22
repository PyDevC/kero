from unittest import TestCase
from kero._engine._kero.dialects import db
from kero._engine._kero.ir import (
    Context,
    Module,
    InsertionPoint,
    Location,
    Type,
    Block
)

from kero._engine._kero._mlir_libs import _keroEngine

class TestOperationGeneration(TestCase):

    def setUp(self):
        self.ctx = Context()
        _keroEngine.register_dialect(self.ctx)
        self.loc = Location.unknown(self.ctx)

    def _make_mod(self):
        return Module.create(self.loc)

    def test_db_types(self):
        mod = self._make_mod()
        table_t = Type.parse('!db.table<"user", 5, 1000>', self.ctx)
        result_t = Type.parse('!db.result', self.ctx)
        assert table_t
        assert result_t
