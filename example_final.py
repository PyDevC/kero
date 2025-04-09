import torch
import pandas as pd
from kero import TableTensor, NumTensor
from kero.engine import Executor, Parser
import numpy as np

id = np.random.randint(1 ,10 , size=(10000000,)) # 10 million queries
age = np.random.randint(18 ,100 , size=(10000000,)) # 10 million queries

# This is how we call 10 million rows and get them queried
device = 'cpu'
table_data = {
    "employee_id": NumTensor(torch.from_numpy(id), name="employee_id"),
    "age": NumTensor(torch.from_numpy(age), name="age"),
    "int": NumTensor(torch.from_numpy(age), name="int"),
    "col": NumTensor(torch.from_numpy(age), name="col"),
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

result = pd.DataFrame(result)
shape = result.shape
pd.set_option('display.max_rows', shape[0])
print(result.head(shape[0]))
