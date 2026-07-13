from unittest import TestCase


class TestMakeDBTypes(TestCase):
    def test_make_dbtable_type_selected_columns(self):
        from kero.engine import _keroEngine, Parser
        from kero._engine._kero import ir
        from kero.engine.codegen import make_dbtable_type
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        operations = parser.parse("SELECT age, salary FROM employee")
        output_op = operations[1]

        ctx = ir.Context()
        _keroEngine.register_dialect(ctx)
        t = make_dbtable_type(output_op.output, ctx)
        self.assertEqual(
            str(t),
            '!db.table<2, 100 : [<"age", i32, 100>, <"salary", i32, 100>]>',
        )

    def test_make_dbtable_type_output_table(self):
        from kero.engine import _keroEngine, Parser
        from kero._engine._kero import ir
        from kero.engine.codegen import make_dbtable_type
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        operations = parser.parse("SELECT spendings FROM employee")
        output_op = operations[1]

        ctx = ir.Context()
        _keroEngine.register_dialect(ctx)
        t = make_dbtable_type(output_op.output, ctx)
        self.assertEqual(
            str(t), '!db.table<1, 100 : [<"spendings", i32, 100>]>'
        )

    def test_make_dbtable_type_star_expansion(self):
        from kero.engine import _keroEngine, Parser
        from kero._engine._kero import ir
        from kero.engine.codegen import make_dbtable_type
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        operations = parser.parse("SELECT * FROM employee")
        scan_op = operations[0]

        ctx = ir.Context()
        _keroEngine.register_dialect(ctx)
        t = make_dbtable_type(scan_op.input, ctx)
        self.assertEqual(
            str(t),
            '!db.table<3, 100 : [<"age", i32, 100>, <"salary", i32, 100>, <"spendings", i32, 100>]>',
        )

    def test_make_dbcolumn_type(self):
        from kero._engine._kero import ir
        from kero.engine import _keroEngine
        from kero.engine.codegen import make_dbcolumn_type

        ctx = ir.Context()
        _keroEngine.register_dialect(ctx)
        t = make_dbcolumn_type("i32", ctx)
        self.assertEqual(str(t), "!db.column<i32>")

    def test_make_dbcolumn_type_i1(self):
        from kero._engine._kero import ir
        from kero.engine import _keroEngine
        from kero.engine.codegen import make_dbcolumn_type

        ctx = ir.Context()
        _keroEngine.register_dialect(ctx)
        t = make_dbcolumn_type("i1", ctx)
        self.assertEqual(str(t), "!db.column<i1>")


class TestIRGen(TestCase):
    def test_select_no_where(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        operations = parser.parse("select salary, spendings from employee")

        irgen = codegen.IRGen("salary_of_employee", operations)
        irgen.emit_ir()

    def test_select_single_column(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("select age from employee")

        irgen = codegen.IRGen("just_age", operations)
        irgen.emit_ir()

    def test_select_star(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("select * from employee")

        irgen = codegen.IRGen("all_columns", operations)
        irgen.emit_ir()

    def test_where_eq(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT age FROM employee WHERE age = 25")

        irgen = codegen.IRGen("age_eq_25", operations)
        irgen.emit_ir()

    def test_where_gt(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT salary FROM employee WHERE salary > 50000")

        irgen = codegen.IRGen("salary_gt_50k", operations)
        irgen.emit_ir()

    def test_where_lt(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse(
            "SELECT spendings FROM employee WHERE spendings < 30000"
        )

        irgen = codegen.IRGen("spendings_lt_30k", operations)
        irgen.emit_ir()

    def test_where_gte(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT age FROM employee WHERE age >= 30")

        irgen = codegen.IRGen("age_gte_30", operations)
        irgen.emit_ir()

    def test_where_lte(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT salary FROM employee WHERE salary <= 40000")

        irgen = codegen.IRGen("salary_lte_40k", operations)
        irgen.emit_ir()

    def test_where_neq(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT age FROM employee WHERE age != 35")

        irgen = codegen.IRGen("age_neq_35", operations)
        irgen.emit_ir()

    def test_multi_column_with_where(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse(
            "SELECT age, salary FROM employee WHERE age > 20"
        )

        irgen = codegen.IRGen("age_salary_filtered", operations)
        irgen.emit_ir()

    def test_different_table_name(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        operations = par.parse("SELECT * FROM employee WHERE salary > 30000")

        irgen = codegen.IRGen("filtered_employee", operations)
        irgen.emit_ir()


class TestAstToKeroConverter(TestCase):
    def setUp(self):
        from kero.engine import _keroEngine
        import kero._engine._kero.ir as ir
        from kero.engine import Parser
        from kero.arrow.samples import all_number_dataset

        self.context = ir.Context()
        self.location = ir.Location.unknown(self.context)
        self.module = ir.Module.create(self.location)

        dataset = all_number_dataset()
        parser = Parser(dataset)
        self.ops_no_where = parser.parse("SELECT age from employee")
        self.ops_with_where = parser.parse(
            "SELECT age FROM employee WHERE age > 10"
        )

        _keroEngine.register_dialect(self.context)

    def generate_func_block(self, operations):
        from kero._engine._kero.dialects import func
        import kero._engine._kero.ir as ir
        from kero.engine.codegen import make_dbtable_type
        from kero.engine.parser.dbast import ScanOp, OutputOp, FilterOp

        first = operations[0]
        last = operations[-1]

        in_t = make_dbtable_type(first.input, self.context)

        out_t = make_dbtable_type(last.output, self.context)

        with self.location, ir.InsertionPoint(self.module.body):
            ftype = ir.FunctionType.get(inputs=[in_t], results=[out_t])
            func_op = func.FuncOp("query", ftype)
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            entry_block = ir.Block.create_at_start(
                func_op.body, ftype.inputs
            )
            return entry_block, ftype

    def test_scan_op_resolves(self):
        from kero.engine.codegen import AstToKeroConverter

        block, _ = self.generate_func_block(self.ops_no_where)
        generator = AstToKeroConverter(
            self.context, self.location, self.module, block
        )
        out = generator.resolve_node(
            self.ops_no_where[0], block.arguments[0]
        )
        self.assertIsNotNone(out)

    def test_scan_output_chain(self):
        from kero.engine.codegen import AstToKeroConverter

        block, _ = self.generate_func_block(self.ops_no_where)
        generator = AstToKeroConverter(
            self.context, self.location, self.module, block
        )
        out = generator.resolve_node(
            self.ops_no_where[0], block.arguments[0]
        )
        out = generator.resolve_node(self.ops_no_where[1], out)
        self.assertIsNotNone(out)

    def test_all_comparison_predicates(self):
        from kero.engine.codegen import AstToKeroConverter
        from kero.engine import Parser
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        cases = [
            ("gt", "SELECT age FROM employee WHERE age > 10"),
            ("gte", "SELECT age FROM employee WHERE age >= 10"),
            ("lt", "SELECT age FROM employee WHERE age < 10"),
            ("lte", "SELECT age FROM employee WHERE age <= 10"),
            ("eq", "SELECT age FROM employee WHERE age = 10"),
            ("neq", "SELECT age FROM employee WHERE age != 10"),
        ]
        for pred, query in cases:
            with self.subTest(predicate=pred):
                ops = parser.parse(query)
                block, _ = self.generate_func_block(ops)
                generator = AstToKeroConverter(self.context, self.location, self.module, block)
                out = block.arguments[0]
                for op in ops:
                    out = generator.resolve_node(op, out)
                self.assertIsNotNone(out)

    def test_output_type_matches_function_result(self):
        from kero.engine.codegen import AstToKeroConverter
        from kero.engine.codegen import make_dbtable_type
        from kero._engine._kero.dialects import func
        import kero._engine._kero.ir as ir
        from kero.engine import Parser
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        parser = Parser(dataset)
        ops = parser.parse(
            "SELECT salary, spendings FROM employee WHERE salary > 50000"
        )

        with self.location, ir.InsertionPoint(self.module.body):
            in_t = make_dbtable_type(ops[0].input, self.context)
            out_t = make_dbtable_type(ops[-1].output, self.context)
            ftype = ir.FunctionType.get(
                inputs=[in_t], results=[out_t]
            )
            func_op = func.FuncOp("query", ftype)
            entry_block = ir.Block.create_at_start(
                func_op.body, ftype.inputs
            )
            generator = AstToKeroConverter(self.context, self.location, self.module, entry_block)
            out = entry_block.arguments[0]
            for op in ops:
                out = generator.resolve_node(op, out)
            self.assertEqual(str(out.type), str(out_t))

    def test_lowering_pipeline(self):
        from kero.engine.codegen import AstToKeroConverter, db_to_llvm_lowering
        import kero._engine._kero.ir as ir
        from kero._engine._kero.dialects import func

        block, _ = self.generate_func_block(self.ops_with_where)
        generator = AstToKeroConverter(self.context, self.location, self.module, block)
        out = block.arguments[0]
        for op in self.ops_with_where:
            out = generator.resolve_node(op, out)
        with self.location, ir.InsertionPoint(block):
            func.return_([out])
        db_to_llvm_lowering(self.module, self.context)


class TestFullPipeline(TestCase):
    def test_select_no_where_roundtrip(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        ops = par.parse("select age from employee")
        irgen = codegen.IRGen("roundtrip", ops)
        irgen.emit_ir()

    def test_select_with_where_roundtrip(self):
        from kero.engine import Parser, codegen
        from kero.arrow.samples import all_number_dataset

        dataset = all_number_dataset()
        par = Parser(dataset)
        ops = par.parse(
            "SELECT age FROM employee WHERE salary > 50000 AND age < 40"
        )
        irgen = codegen.IRGen("roundtrip_filter", ops)
        irgen.emit_ir()
