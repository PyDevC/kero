import kero
import string
import random
import numpy as np
import torch
from kero import QueryFilter
from kero.engine import Executor
from psycopg_api.database import DatabaseAPI
from kero import TableTensor, NumTensor

employee_id = np.random.randint(1000, 10000, size=(1000,))
age = np.random.randint(1,100,size=(1000,))

columns = {
    "employee_id": NumTensor(torch.tensor(employee_id), "employee_id"),
    "age": NumTensor(torch.tensor(age), "age"),
}
query = "SELECT employee_id FROM employees WHERE age > 10"

kquery = {
    "columns": columns.keys(),
    "where": {
        "operator": ">",
        "left": "age",
        "right": "10"
    }
}

employees = TableTensor(columns, "employees")
executor = Executor(employees)
result = executor.execute_query(kquery)
print(result)
