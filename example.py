import kero.engine as engine
from kero.tensors import NumTensor, TableTensor
import torch
import pandas as pd

num1 = NumTensor(torch.randn(111,), "num1")
num2 = NumTensor(torch.randn(111,), "num2")
num3 = NumTensor(torch.randn(111,), "num3")

table = TableTensor({"num1": num1, "num2": num2, "num3": num3}, "Main")

parser = engine.Parser()
query = "SELECT num1, num2 FROM Main WHERE num1 < 10"
qkey = parser.parse(query)

executor = engine.Executor(table)
out = executor.execute(qkey)

df = pd.DataFrame(out)
print(df.head())
