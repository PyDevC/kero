import torch
from kero import TableTensor
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
                select_data[c[0]] = torch.masked_select(c[1].tensor, mask)
            self.result = select_data
        
        return self.result

    def _select_columns(self, mask: torch.BoolTensor, columns: List[str]) -> torch.Tensor:
        temp = {key: self.data.columns[key] for key in columns}
        return temp
