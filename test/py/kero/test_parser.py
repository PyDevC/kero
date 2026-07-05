from unittest import TestCase

from kero.arrow import samples as sample
from kero.arrow.data import Dataset
from kero.arrow.exce import TableNotFoundException
from kero.engine.parser import Parser, GlotToDB
from kero.engine.parser.dbast import ScanOp, OutputOp, FilterOp, CmpIOp
from kero.engine.parser.parser import GlotConversionNotPossible, NodeNotFound, NodeNotImplemented


class TestGlotToDB(TestCase):
    def test_select_star(self):
        converter = GlotToDB("SELECT * FROM person")
        result = converter.convert()
        self.assertEqual(len(result), 2)
        scan, output = result
        self.assertIsInstance(scan, ScanOp)
        self.assertIsInstance(output, OutputOp)
        self.assertEqual(scan.table.metadata.metadata["name"], "person")
        self.assertTrue(scan.table.columns[0].metadata.metadata["is_star"])

    def test_select_columns(self):
        converter = GlotToDB("SELECT age, name FROM person")
        result = converter.convert()
        scan, output = result
        self.assertEqual(len(result), 2)
        col_names = [c.metadata.metadata["name"] for c in scan.table.columns]
        self.assertEqual(col_names, ["age", "name"])

    def test_select_star_with_where(self):
        converter = GlotToDB("SELECT * FROM person WHERE age > 10")
        result = converter.convert()
        self.assertEqual(len(result), 3)
        scan, filter_op, output = result
        self.assertIsInstance(filter_op, FilterOp)

        cmp_op = filter_op.region.operations
        self.assertIsInstance(cmp_op, CmpIOp)
        self.assertEqual(cmp_op.predicate, "gt")
        self.assertEqual(cmp_op.lhs.name, "age")
        self.assertEqual(cmp_op.rhs.constant, 10)

    def test_select_columns_with_where(self):
        converter = GlotToDB("SELECT name FROM person WHERE age = 20")
        result = converter.convert()
        self.assertEqual(len(result), 3)
        scan, filter_op, output = result

        col_names = [c.metadata.metadata["name"] for c in scan.table.columns]
        self.assertEqual(col_names, ["name"])

        cmp_op = filter_op.region.operations
        self.assertEqual(cmp_op.predicate, "eq")
        self.assertEqual(cmp_op.lhs.name, "age")
        self.assertEqual(cmp_op.rhs.constant, 20)

    def test_all_comparison_ops(self):
        cases = [
            (">", "gt"),
            (">=", "gte"),
            ("<", "lt"),
            ("<=", "lte"),
            ("=", "eq"),
            ("!=", "neq"),
        ]
        for op_str, pred in cases:
            query = f"SELECT * FROM person WHERE age {op_str} 5"
            converter = GlotToDB(query)
            result = converter.convert()
            _, filter_op, _ = result
            cmp_op = filter_op.region.operations
            self.assertEqual(cmp_op.predicate, pred, f"failed for {op_str}")

    def test_output_select_matches_columns(self):
        converter = GlotToDB("SELECT age, name FROM person")
        result = converter.convert()
        _, output = result
        self.assertEqual(output.select, ["age", "name"])

    def test_output_table_is_copy_of_input(self):
        converter = GlotToDB("SELECT age FROM person")
        result = converter.convert()
        scan, output = result
        self.assertIs(output.input, scan.table)
        self.assertIsNotNone(output.output)

    def test_no_table_raises(self):
        from sqlglot import exp
        converter = GlotToDB("SELECT age FROM person")
        select = converter.parsed_query.find(exp.Select)
        select.args["from_"] = None
        with self.assertRaises(NodeNotFound):
            converter.convert()


class TestGlotToDBErrors(TestCase):
    def test_and_raises_not_implemented(self):
        with self.assertRaises(NodeNotImplemented):
            GlotToDB("SELECT * FROM person WHERE age > 10 AND age < 20").convert()

    def test_or_raises_not_implemented(self):
        with self.assertRaises(NodeNotImplemented):
            GlotToDB("SELECT * FROM person WHERE age > 10 OR age < 20").convert()

    def test_where_literal_raises_conversion(self):
        with self.assertRaises(GlotConversionNotPossible):
            GlotToDB("SELECT * FROM person WHERE 1").convert()

    def test_where_column_only_raises_conversion(self):
        with self.assertRaises(GlotConversionNotPossible):
            GlotToDB("SELECT * FROM person WHERE age").convert()


class TestParserFullPipeline(TestCase):
    def test_parse_select_star_resolves_types(self):
        dataset = sample.all_number_dataset()
        parser = Parser(dataset)
        result = parser.parse("SELECT * FROM employee")
        self.assertEqual(len(result), 2)
        scan, output = result
        for col in scan.table.columns:
            self.assertIsNotNone(col.metadata.metadata["dtype"])
            self.assertEqual(
                col.metadata.metadata["nrows"],
                dataset.tables["employee"].num_rows,
            )

    def test_parse_select_columns_resolves_types(self):
        dataset = sample.all_number_dataset()
        parser = Parser(dataset)
        result = parser.parse("SELECT age, salary FROM employee")
        self.assertEqual(len(result), 2)
        scan, output = result
        self.assertEqual(len(scan.table.columns), 2)
        for col in scan.table.columns:
            self.assertEqual(col.metadata.metadata["dtype"], "i32")
            self.assertEqual(col.metadata.metadata["nrows"], 100)

    def test_parse_with_where_resolves_types(self):
        dataset = sample.all_number_dataset()
        parser = Parser(dataset)
        result = parser.parse("SELECT age FROM employee WHERE age > 10")
        self.assertEqual(len(result), 3)
        scan, filter_op, output = result

        self.assertEqual(scan.table.columns[0].metadata.metadata["dtype"], "i32")
        cmp_op = filter_op.region.operations
        self.assertEqual(cmp_op.lhs.dtype, "i32")
        self.assertEqual(cmp_op.output.dtype, "bool")

    def test_filter_output_table_resolved(self):
        dataset = sample.all_number_dataset()
        parser = Parser(dataset)
        result = parser.parse("SELECT age, salary FROM employee WHERE age > 10")
        _, filter_op, output = result

        for col_attr in output.output.columns:
            self.assertIsNotNone(col_attr.metadata.metadata["dtype"])

        for col_attr in filter_op.output.columns:
            self.assertIsNotNone(col_attr.metadata.metadata["dtype"])

    def test_toy_school_types(self):
        dataset = sample.toy_school_dataset()
        parser = Parser(dataset)
        result = parser.parse("SELECT age FROM person WHERE age > 10")
        scan, filter_op, output = result

        age_col = scan.table.columns[0]
        self.assertEqual(age_col.metadata.metadata["dtype"], "i8")
        self.assertEqual(age_col.metadata.metadata["nrows"], 5)

    def test_unknown_table_raises(self):
        dataset = Dataset()
        parser = Parser(dataset)
        with self.assertRaises(TableNotFoundException):
            parser.parse("SELECT * FROM nonexistent")
