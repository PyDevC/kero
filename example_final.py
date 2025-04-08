import torch
from kero import TableTensor, NumTensor
from kero.engine import Executor, Parser
import numpy as np

id = np.asarray([1,1,1,1,1,2,1,1,1,1])
age = np.random.randint(18 ,100 , size=(10,1))

table_data = {
    "employee_id": NumTensor(torch.from_numpy(id), name="employee_id"),
    "age": NumTensor(torch.from_numpy(age), name="age")
}
table_tensor = TableTensor(columns=table_data, name="employees")

query = """
FROM employees 
WHERE employee_id > 1; 
"""

parser = Parser()

kquery = parser.parse(query)

executor = Executor(table_tensor)

result = executor.execute(kquery)
print("Final Result:", result)
