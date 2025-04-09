import torch
from typing import Dict, List, Union
from kero import TableTensor
from kero.engine.operations.operators import eq, gt, lt, ge, le, ne, add, sub, prod, div

class KeroCompiler:
    """Compiler Executes the code and generates a mask for torch.select_masked
    A compiler instance is only available for the single TensorTable
    """
    def __init__(self, table: TableTensor):
        self.table = table

    def compile(self, operations: List[Dict[str, Union[str, int]]]) -> torch.Tensor:
        """gets the operations from kquery to get the results
        """
        result_tensor = None

        for operation in operations:
            operator = operation["operator"]
            left_operand = self._get_tensor(operation["left"])
            right_operand = self._get_tensor(operation["right"])

            # Execute based on operator type
            result_tensor = self._execute_operation(operator, left_operand, right_operand)

        return result_tensor

    def _get_tensor(self, operand: Union[str, int]) -> torch.Tensor:
        """gets the tensors from the operands in the kquery
        """
        try:
            operand = int(operand)
        except:
            pass
        if isinstance(operand, str):
            # Operand is a column name
            return self.table.columns[operand].tensor
        # Operand is a constant value
        return torch.tensor([operand])

    def _execute_operation(self, operator: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """executes the tensor operation
        """
        if not isinstance(right, torch.Tensor):
            right = torch.tensor(right)
        if not isinstance(left, torch.Tensor):
            right = torch.tensor(left)

        if operator == "=":
            mask = eq(left, right).execute()
            return mask
        elif operator == "!=":
            mask = ne(left, right).execute()
            return mask
        elif operator == ">":
            mask = gt(left, right).execute()
            return mask
        elif operator == "<":
            mask = lt(left, right).execute()
            return mask
        elif operator == ">=":
            mask = ge(left, right).execute()
            return mask
        elif operator == "<=":
            mask = le(left, right).execute()
            return mask
        
        raise ValueError(f"Unsupported operator: {operator}")
