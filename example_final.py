import torch
from kero import TableTensor, NumTensor
from kero.engine import Executor, Parser
import numpy as np

id = np.random.randint(1 ,10 , size=(1000,)) # 10 million queries
age = np.random.randint(18 ,100 , size=(1000,)) # 10 million queries

device = 'cuda' if torch.cuda.is_available() else 'cpu'

table_data = {
    "employee_id": NumTensor(torch.from_numpy(id), name="employee_id"),
    "age": NumTensor(torch.from_numpy(age), name="age")
}
table_tensor = TableTensor(columns=table_data, name="employees")
table_tensor.to(device)

query = """
SELECT employee_id, age
FROM employees 
WHERE employee_id > 7; 
"""

parser = Parser()

kquery = parser.parse(query)

executor = Executor(table_tensor)

result = executor.execute(kquery)
print(result)
