from unittest import TestCase

class TestTypeGeneration(TestCase):
    def setUp(self):
        from kero.engine import _keroEngine
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        self.context = ir.Context()
        self.location = ir.Location.unknown(self.context)
        self.module = ir.Module.create(self.location)

        _keroEngine.register_dialect(self.context)

    def test_table_type_gen(self):
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        with self.location, ir.InsertionPoint(self.module.body):
            table_t = ir.Type.parse('!db.table<1, 100 : [#db.column<"age", i32, 100>]>')

class TestOpGeneartion(TestCase):
    def setUp(self):
        from kero.engine import _keroEngine
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        self.context = ir.Context()
        self.location = ir.Location.unknown(self.context)
        self.module = ir.Module.create(self.location)

        _keroEngine.register_dialect(self.context)

    def test_scan_op_gen(self):
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        with self.location, ir.InsertionPoint(self.module.body):
            table_t = ir.Type.parse('!db.table<1, 100 : [#db.column<"age", i32, 100>]>')

            ftype = ir.FunctionType.get(inputs=[table_t], results=[table_t])
            
            func_op = func.FuncOp("query", ftype)

            entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
            with ir.InsertionPoint(entry_block):
                scan_op = db.scan(output=table_t, table=entry_block.arguments[0])

    def test_output_op_gen(self):
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        with self.location, ir.InsertionPoint(self.module.body):
            table_t = ir.Type.parse('!db.table<1, 100 : [#db.column<"age", i32, 100>]>')

            ftype = ir.FunctionType.get(inputs=[table_t], results=[table_t])
            
            func_op = func.FuncOp("query", ftype)

            entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
            with ir.InsertionPoint(entry_block):
                scan_op = db.scan(output=table_t, table=entry_block.arguments[0])
                output_op = db.output(output=table_t, table=scan_op, select=["age"])
                func.return_([output_op])
