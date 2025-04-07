import torch
from typing import Dict, List, Union
from kero.tensors import TableTensor, NumTensor, StrTensor, DateTensor
from kero.engine.operations.operators import lt, eq, gt

class KeroCompiler:
    """
    A compiler that transforms intermediate relational query representations (kquery)
    into tensor operations and executes them on a TableTensor.
    """
    def __init__(self, table: TableTensor):
        self.table = table

    def compile(self, kquery: Dict) -> torch.Tensor:
        """
        Compile and execute the kquery on the given TableTensor.

        Args:
            kquery (Dict): Intermediate representation of the query.

        Returns:
            torch.Tensor: Resulting tensor after executing the query.
        """
        # Step 1: Process WHERE clause for filtering
        mask = self._process_where(kquery.get('where')) if 'where' in kquery else None

        # Step 2: Select columns based on SELECT clause
        selected_columns = kquery.get('columns', list(self.table.columns.keys()))
        tensors = []

        for col_name in selected_columns:
            tensor = self.table.columns[col_name].tensor
            if mask is not None:
                tensor = tensor[mask]
            tensors.append(tensor)

        # Step 3: Stack tensors to form the final result
        result = torch.stack(tensors, dim=1) if tensors else torch.tensor([])
        return result

    def _process_where(self, condition: Dict) -> torch.BoolTensor:
        """
        Recursively process the WHERE clause to generate a boolean mask.

        Args:
            condition (Dict): Condition dictionary representing the WHERE clause.

        Returns:
            torch.BoolTensor: Boolean mask for filtering rows.
        """
        operator = condition['operator']

        if operator == '=':
            left_tensor = self._get_operand(condition['left'])
            right_tensor = self._get_operand(condition['right'])
            return eq(left_tensor, right_tensor).execute()

        elif operator == '>':
            left_tensor = self._get_operand(condition['left'])
            right_tensor = self._get_operand(condition['right'])
            return gt(left_tensor, right_tensor).execute()

        elif operator == '<':
            left_tensor = self._get_operand(condition['left'])
            right_tensor = self._get_operand(condition['right'])
            return lt(left_tensor, right_tensor).execute()

        raise ValueError(f"Unsupported operator: {operator}")

    def _get_operand(self, operand: Union[str, int, float]) -> torch.Tensor:
        """
        Resolve an operand to its corresponding tensor.

        Args:
            operand (Union[str, int, float]): Operand value or column name.

        Returns:
            torch.Tensor: Corresponding tensor.
        """
        if isinstance(operand, str):
            # Operand is a column name
            return self.table.columns[operand].tensor
        # Operand is a constant value
        return torch.tensor(operand)
