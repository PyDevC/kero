import torch
from typing import Dict, List, Union
from kero import TableTensor
from kero.engine.operations.operators import (
    EqOp as eq, GtOp as gt, LtOp as lt, GeOp as ge, LeOp as le, NeOp as ne,
    AddOp as add, SubOp as sub, ProdOp as prod, DivOp as div,
)
from kero.engine.operations.operators import Operator, BinaryOperator, UnaryOperator


class OperatorRegistry:
    _operators: Dict[str, type] = {
        "=": EqOp,
        "!=": NeOp,
        ">": GtOp,
        "<": LtOp,
        ">=": GeOp,
        "<=": LeOp,
        "+": AddOp,
        "-": SubOp,
        "*": ProdOp,
        "/": DivOp,
    }

    @classmethod
    def get_operator(cls, symbol: str) -> type:
        if symbol not in cls._operators:
            raise ValueError(f"Unsupported operator: {symbol}")
        return cls._operators[symbol]

    @classmethod
    def register(cls, symbol: str, operator_class: type):
        if not issubclass(operator_class, BinaryOperator) and not issubclass(operator_class, UnaryOperator):
            raise TypeError("Operator must be a BinaryOperator or UnaryOperator")
        cls._operators[symbol] = operator_class


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
        if isinstance(operand, int):
            return torch.tensor([operand])
        if isinstance(operand, str):
            try:
                return self.table.columns[operand].tensor
            except KeyError:
                raise KeyError(f"Column '{operand}' not found in table '{self.table.name}'")
        return torch.tensor([operand])

    def _execute_operation(self, operator: str, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """executes the tensor operation
        """
        if not isinstance(right, torch.Tensor):
            right = torch.tensor(right)
        if not isinstance(left, torch.Tensor):
            right = torch.tensor(left)

        operator_class = OperatorRegistry.get_operator(operator)
        return operator_class(left, right).execute()
