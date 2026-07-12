from kero.arrow import samples as sample
from kero.engine.parser import Parser, GlotToDB

from unittest import TestCase


class TestGlotToDB(TestCase):
    def setUp(self):
        self.dataset = sample.all_number_dataset()

    def test_select_star(self):
        converter = GlotToDB("SELECT * FROM employee", self.dataset)
        result = converter.convert()
        ScanOp = result[0]
        self.assertEqual(ScanOp.input.metadata.metadata["name"], "employee")
        OutputOp = result[1]
        output_column_names = set(OutputOp.output.metadata.metadata["column_names"])
        input_column_names = set(OutputOp.input.metadata.metadata["column_names"])
        self.assertSetEqual(output_column_names.intersection(input_column_names), input_column_names)

    def test_select_columns(self):
        converter = GlotToDB("SELECT age, salary FROM employee", self.dataset)
        result = converter.convert()
        OutputOp = result[1]

        output_column_names = set(OutputOp.output.metadata.metadata["column_names"])
        input_column_names = set(OutputOp.input.metadata.metadata["column_names"])
        
        self.assertSetEqual(output_column_names.intersection(input_column_names), set(("age", "salary")))

    def test_select_star_with_where(self):
        converter = GlotToDB("SELECT * FROM employee WHERE age > 20", self.dataset)
        result = converter.convert()
        
        FilterOp = result[1]
        OutputOp = result[2]

    def test_select_columns_with_where(self):
        converter = GlotToDB("SELECT salary FROM employee WHERE age > 20", self.dataset)
        result = converter.convert()
        
        FilterOp = result[1]
        assert FilterOp
        OutputOp = result[2]
        assert OutputOp

    def test_select_columns_with_where_and(self):
        converter = GlotToDB("SELECT salary FROM employee WHERE age > 20 and salary > 10000", self.dataset)
        result = converter.convert()
        
        FilterOp = result[1]
        assert FilterOp
        OutputOp = result[2]
        assert OutputOp
