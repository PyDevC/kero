from unittest import TestCase

from kero._engine._kero._mlir_libs import _keroEngine
from kero._engine._kero.dialects import arith, db, func
from kero._engine._kero.passmanager import PassManager
import kero._engine._kero.ir as ir


class TestOperationLowering(TestCase):

    def setUp(self):
        self.ctx = ir.Context()
        _keroEngine.register_dialect(self.ctx)
        self.loc = ir.Location.unknown(self.ctx)
        self.module = ir.Module.create(loc=self.loc)

    def _apply_lowering(self):
        with self.ctx:
            pm = PassManager.parse('builtin.module(db-to-tensor)')
            pm.run(self.module.operation)
        return str(self.module)

    def test_lowering_of_types(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                table_t = ir.Type.parse('!db.table<"user", 100, 1000>')
                column_t = ir.Type.parse('!db.column<"user", "id", i32, 1000>')

                @func.FuncOp.from_py_func(table_t, column_t)
                def test_types(arg0, arg1):
                    return

        output = self._apply_lowering()
        assert 'tensor<' in output
        assert '!db.' not in output

    def test_lowering_of_scan(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                table_t = ir.Type.parse('!db.table<"user", 100, 1000>')
                result_t = ir.Type.parse('!db.result')

                @func.FuncOp.from_py_func(table_t)
                def test_scan(arg0):
                    scan_val = db.scan(result=result_t, table=arg0)
                    return scan_val

        output = self._apply_lowering()
        assert 'tensor<' in output
        assert 'db.scan' not in output

    def test_lowering_of_filter(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                table_t = ir.Type.parse('!db.table<"user", 100, 1000>')
                column_t = ir.Type.parse('!db.column<"user", "age", i32, 1000>')
                result_t = ir.Type.parse('!db.result')
                row_t = ir.Type.parse('!db.row')
                i32_t = ir.Type.parse('i32')


                @func.FuncOp.from_py_func(table_t, column_t)
                def test_db_filter_op(arg0, arg1):
                    scan_op = db.scan(
                        result=result_t,
                        table=arg0,
                    )

                    filter_op = db.FilterOp(
                        output=result_t,
                        input=scan_op,
                    )

                    region = filter_op.region
                    block = region.blocks.append(row_t)

                    with ir.InsertionPoint(block):
                        age_val = db.getcol(
                            val=i32_t,
                            row=block.arguments[0],
                            column=arg1,
                        )

                        age_limit = arith.constant(i32_t, 10)

                        cond = arith.cmpi(arith.CmpIPredicate.sgt, age_val, age_limit)

                        db.return_(cond)

                    return filter_op.result

        output = self._apply_lowering()
        assert 'linalg.generic' in output
        assert 'db.filter' not in output

    def test_lowering_of_getcol(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                column_t = ir.Type.parse('!db.column<"user", "age", i32, 1000>')
                row_t = ir.Type.parse('!db.row')
                i32_t = ir.Type.parse('i32')

                @func.FuncOp.from_py_func(row_t, column_t)
                def test_getcol(arg0, arg1):
                    val = db.getcol(val=i32_t, row=arg0, column=arg1)
                    return val

        output = self._apply_lowering()
        assert 'tensor.extract' in output
        assert 'db.getcol' not in output
