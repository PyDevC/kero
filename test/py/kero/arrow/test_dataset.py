from unittest import TestCase

import pyarrow as pa
import numpy as np
import kero.arrow.data as d

class TestDataset(TestCase):
    def test_dataset_creation(self):
        table_size = 100
        age = np.random.randint(18, 90, size=table_size, dtype=np.int32)
        salary = np.random.random_sample(table_size)
        budget = np.random.randint(1000, 100000, size=table_size, dtype=np.int32)

        employee = {
            "age": age,
            "salary": salary,
            "budget": budget 
        }

        employee_table = pa.Table.from_pydict(employee)

        tables = {
            "employee": employee_table
        }

        dataset = d.Dataset(tables)
        assert(len(dataset) == 1)
        assert(dataset.get_table("employee"))
        n_age, n_salary, n_budget = dataset.get_table_as_arrays("employee")
        self.assertTrue(np.array_equal(n_age, age))
        self.assertTrue(np.array_equal(n_salary, salary))
        self.assertTrue(np.array_equal(n_budget, budget))
