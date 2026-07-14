import time
import os
from unittest import TestCase


SIZES = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)
queries = (
    "SELECT age, salary, spendings FROM employee",
    "SELECT * FROM employee WHERE age > 50",
    "SELECT salary FROM employee WHERE salary <= 40000 AND age < 12",
    "SELECT * FROM employee WHERE age > 20 AND salary < 100000 OR salary = 10000 AND department_id = 1 OR department_id = 2 AND experience_years >= 5 AND performance_rating != 3 OR is_active = 1 AND hire_year <= 2025 AND bonus_eligible = 1 OR manager_id = 101 AND age < 30 OR NOT region_id = 4 AND termination_year = 0 OR position_level > 2 AND certification_count >= 1"
)

class TestBenchmark(TestCase):
    def test_benchmark(self):
        from kero.engine import Parser, codegen
        from kero.engine.execution import KeroEngine
        from kero.arrow.samples import employee_table

        perf_matrix: list[list[dict[str, float]]] = []

        for size in SIZES:
            dataset = employee_table(size=size)
            parser = Parser(dataset)

            per_query_time: list[dict[str, float]] = []

            for query in queries:
                start_parsing_time = time.perf_counter()
                query_ast = parser.parse(query)
                end_parsing_time = time.perf_counter()

                start_codegen_time = time.perf_counter()
                irgen = codegen.IRGen("q", query_ast)
                irgen.emit_ir()
                end_codegen_time = time.perf_counter()

                start_lowering_time = time.perf_counter()
                codegen.db_to_llvm_lowering(irgen.module, irgen.context)
                end_lowering_time = time.perf_counter()

                engine = KeroEngine(irgen.module, irgen.context)
                engine.configure_outputs([i for i in range(irgen.func_result_num)])
                start_execution_time = time.perf_counter()
                results = engine.execute("q", dataset, ["employee"])
                end_execution_time = time.perf_counter()

                array = engine.results_to_numpy(results)
                assert array

                del irgen

                performance = {
                    "parsing": (end_parsing_time - start_parsing_time),
                    "codegen": (end_codegen_time - start_codegen_time),
                    "lowering": (end_lowering_time - start_lowering_time),
                    "execution": (end_execution_time - start_execution_time)
                }

                per_query_time.append(performance)

            perf_matrix.append(per_query_time)

        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "Output", 
            "performance.txt"
        )

        with open(file_path, "a") as file:
            for perf_m, size in zip(perf_matrix, SIZES):
                file.write(f"For Size: {size}\n")
                for per_query, query in zip(perf_m, queries):
                    file.write(f"\tFor Query: {query}\n\t\t")
                    file.write(repr(per_query))
                    file.write("\n")

            file.write("-"*80)
            file.write("\n")
            file.write("-"*80)
            file.write("\n")
            file.close()
