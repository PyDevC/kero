from unittest import TestCase

class TestIRGen(TestCase):
    def test_ir_gen(self):
        from kero.engine import codegen, parser
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = parser.Parser(dataset)
        operations = par.parse("select salary, spendings from employee")

        irgen = codegen.IRGen("salary_of_person", operations)
        irgen.emit_ir()


class TestAstToKeroConverter(TestCase):
    def setUp(self):
        from kero.engine import _keroEngine
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db
        from kero.engine.parser import Parser
        from kero.arrow.samples import all_number_dataset

        self.context = ir.Context()
        self.location = ir.Location.unknown(self.context)
        self.module = ir.Module.create(self.location)

        dataset = all_number_dataset()
        parser = Parser(dataset)
        self.operations = parser.parse("SELECT age, salary from employee")

        _keroEngine.register_dialect(self.context)

    def generate_func_block(self):
        from kero._engine._kero.dialects import func, db
        import kero._engine._kero.ir as ir
        from kero.arrow.type_resolve import PYARROW_TO_DB_TYPES
        from kero.engine.codegen import make_dbtable_type

        import pyarrow as pa

        with self.location, ir.InsertionPoint(self.module.body):
            table_t = make_dbtable_type(self.operations[0].input_table, self.context)
            output_t = make_dbtable_type(self.operations[-1].output_table, self.context)
            ftype = ir.FunctionType.get(inputs=[table_t], results=[output_t])

            func_op = func.FuncOp("query", ftype)
            entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
            return entry_block

    def test_ast_to_kero_type_gen(self):
        from kero.engine.codegen import AstToKeroConverter
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func, db

        block = self.generate_func_block()

        generator = AstToKeroConverter(self.context, 
                                       self.location,
                                       self.module,
                                       block)

        out = block.arguments[0]
        with self.location, ir.InsertionPoint(block):
            for operation in self.operations:
                out = generator.resolve_node(operation, out)

            func.return_([out])
