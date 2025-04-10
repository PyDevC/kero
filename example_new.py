import torch
from kero import NumTensor, StrTensor, TableTensor
from kero.engine import Executor, Parser
import pandas as pd

path = '../../documents/assignments/ds/data/retail_data_analysis/sales data-set.csv'
data = pd.read_csv(path)

columns = {
    data.columns[0]: NumTensor(torch.tensor(data[data.columns[0]], dtype=torch.uint8), name=data.columns[0]),
    data.columns[1]: NumTensor(torch.tensor(data[data.columns[1]], dtype=torch.uint8), name=data.columns[0]),
    data.columns[3]: NumTensor(torch.tensor(data[data.columns[3]], dtype=torch.float64), name=data.columns[3]),
    data.columns[4]: NumTensor(torch.tensor(data[data.columns[4]], dtype=torch.bool), name=data.columns[4]),
}

table = TableTensor(columns, name="sales")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = Parser()
execut = Executor(table)

while True:
    query = input("Input your query: ")
    kquery = parser.parse(query)
    result = execut.execute(kquery)
    result = pd.DataFrame(result)
    shape = result.shape
    pd.set_option('display.max_rows', shape[0])
    print(result.head(shape[0]))
