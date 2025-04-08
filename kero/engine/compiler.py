import torch
from typing import Dict, List, Union
from kero import TableTensor
from kero.engine.operations.operators import eq, gt, lt, ge, le, ne, add, sub, prod, div

class KeroCompiler:
    """
    A compiler that processes individual operations from Intermediate Representation (IR)
    and applies them to TableTensor data.
    """

    def __init__(self, table: TableTensor):
        """
        Initialize the compiler with a TableTensor.

        Args:
            table (TableTensor): The tensor-based representation of a relational table.
        """
        self.table = table

    def compile(self, operations: List[Dict[str, Union[str, int]]]) -> torch.Tensor:
        """
        Compile and execute a series of operations on the TableTensor.

        Args:
            operations (List[Dict[str, Union[str, int]]]): List of operations from IR.

        Returns:
            torch.Tensor: Resulting tensor after executing all operations.
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
        """
        Resolve an operand to its corresponding tensor.

        Args:
            operand (Union[str, int]): Operand value or column name.

        Returns:
            torch.Tensor: Corresponding tensor.
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
        """
        Execute an operation based on its operator type.

        Args:
            operator (str): Operator type (e.g., '=', '>', '<').
            left (torch.Tensor): Left operand tensor.
            right (torch.Tensor): Right operand tensor.

        Returns:
            torch.Tensor: Resulting tensor after applying the operation.
        """
        if operator == "=":
            return eq(left, right).execute()
        elif operator == "!=":
            return ne(left, right).execute()
        elif operator == ">":
            return gt(left, right).execute()
        elif operator == "<":
            return lt(left, right).execute()
        elif operator == ">=":
            return ge(left, right).execute()
        elif operator == "<=":
            return le(left, right).execute()
        
        raise ValueError(f"Unsupported operator: {operator}")
