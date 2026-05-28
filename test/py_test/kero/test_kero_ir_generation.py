from unittest import TestCase

from kero._engine._kero._mlir_libs import _keroEngine
from kero._engine._kero.dialects import db, func
import kero._engine._kero.ir as ir

class TestOperationGeneration(TestCase):

    def setUp(self):
        self.ctx = ir.Context()
        _keroEngine.register_dialect(self.ctx)
        self.loc = ir.Location.unknown(self.ctx)
        self.module = ir.Module.create(loc=self.loc)
        self.func = None

    def test_db_types(self):
        with self.ctx, self.loc:
            table_t = ir.Type.parse('!db.table<"user", 5, 1000>')
            column_t = ir.Type.parse(f'!db.column<"user", "age", i32, 1000>')
            result_t = ir.Type.parse('!db.result')

            assert isinstance(table_t, ir.Type)
            assert isinstance(column_t, ir.Type)
            assert isinstance(result_t, ir.Type)

    def test_scan_op(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                table_t = ir.Type.parse('!db.table<"user", 5, 1000>')
                result_t = ir.Type.parse('!db.result')


                @func.FuncOp.from_py_func(table_t)
                def test_scan_op_func(arg0):
                    scanop = db.scan(
                        result=result_t,
                        table=arg0,
                    )

                    return scanop
