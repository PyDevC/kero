from kero.arrow.samples import all_number_dataset, employee_table
from kero.engine import Parser, codegen
from kero.engine.execution import KeroEngine

from unittest import TestCase

class TestExecutionPipeline(TestCase):
    def setUp(self):
        self.dataset = all_number_dataset(size=10_000)
        self.parser = Parser(self.dataset)

    def test_get_aged_employee(self):
        query = "SELECT * FROM employee WHERE age > 20 AND salary < 100000 OR salary = 10000"
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

    def test_complex_query(self):
        query = """
        SELECT * FROM employee 
        WHERE age > 20 AND 
            salary < 100000 OR 
            salary = 10000 AND 
            department_id = 1 OR 
            department_id = 2 AND 
            experience_years >= 5 AND 
            performance_rating != 3 OR 
            is_active = 1 AND 
            hire_year <= 2025 AND 
            bonus_eligible = 1 OR 
            manager_id = 101 AND 
            age < 30 OR NOT 
            region_id = 4 AND 
            termination_year = 0 OR 
            position_level > 2 AND 
            certification_count >= 1
        """

        dataset = employee_table()
        parser = Parser(dataset)
        query_ast = parser.parse(query)
        irgen = codegen.IRGen("employee_complex_sheet", query_ast)
        irgen.emit_ir()
        codegen.db_to_llvm_lowering(irgen.module, irgen.context)

        exe = KeroEngine(irgen.module, irgen.context)
        exe.configure_outputs([i for i in range(irgen.func_result_num)])

        results = exe.execute("employee_complex_sheet", dataset, ["employee"])
        numpy_results = exe.results_to_numpy(results)
        for id, array in numpy_results.items():
            assert(array.size)
