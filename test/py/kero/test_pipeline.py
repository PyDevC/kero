from kero.arrow.samples import all_number_dataset
from kero.engine import Parser, codegen
from kero.engine.execution import KeroEngine

from unittest import TestCase

class TestExecutionPipeline(TestCase):
    def setUp(self):
        self.dataset = all_number_dataset(size=10_000)
        self.parser = Parser(self.dataset)

    def test_get_aged_employee(self):
        query = "SELECT * FROM employee WHERE age > 20"
        query_ast = self.parser.parse(query)
        irgen = codegen.IRGen("get_aged_employee", query_ast)
        irgen.emit_ir()
        codegen.db_to_llvm_lowering(irgen.module, irgen.context)

        exe = KeroEngine(irgen.module, irgen.context)
        exe.configure_outputs([i for i in range(irgen.func_result_num)])

        results = exe.execute("get_aged_employee", self.dataset, ["employee"])
        numpy_results = exe.results_to_numpy(results)
        for id, array in numpy_results.items():
            assert(array.size)
