import torch
from kero import TableTensor, StrTensor
from kero.engine import KeroCompiler
from typing import Dict, List, Any

class Executor:
    def __init__(self, data: TableTensor):
        self.data = data
        self.compiler = KeroCompiler(data)
        self.result = None

    def execute(self, kquery: Dict[str, Any]) -> torch.Tensor:
        mask = self.compiler.compile(kquery['operations'])
        
        if 'columns' in kquery:
            selected_columns = kquery['columns']
            cols = self._select_columns(mask, selected_columns)
            select_data = {}
            for c in cols.items():
                if isinstance(c[1], StrTensor):
                    c[1].tensor = c[1].tensor[mask]
                    select_data[c[0]] = [tensor_to_string(row) for row in c[1].tensor]
                else:
                    select_data[c[0]] = torch.masked_select(c[1].tensor, mask)
            self.result = select_data
        
        return self.result

    def _select_columns(self, mask: torch.BoolTensor, columns: List[str]) -> torch.Tensor:
        temp = {key: self.data.columns[key] for key in columns}
        return temp

def tensor_to_string(tensor):
    return ''.join([chr(x.item()) for x in tensor if x.item() != 0])
