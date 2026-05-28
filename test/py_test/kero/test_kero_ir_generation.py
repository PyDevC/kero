from unittest import TestCase

from kero._engine._kero._mlir_libs import _keroEngine
from kero._engine._kero.dialects import arith, db, func
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
                    scan_op = db.scan(
                        result=result_t,
                        table=arg0,
                    )

                    return scan_op

    def test_filter_op(self):
        with self.ctx, self.loc:
            with ir.InsertionPoint(self.module.body):
                table_t = ir.Type.parse('!db.table<"user", 10, 100>')
                column_t = ir.Type.parse(f'!db.column<"user", "age", i32, 100>')
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
