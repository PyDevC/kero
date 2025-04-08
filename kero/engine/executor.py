import torch
from kero import TableTensor
from kero.engine import KeroCompiler
from typing import Dict, List, Any

class Executor:
    """
    An executor that initializes with data, executes a kquery using tensor operations,
    and outputs the final data after all operations.
    """

    def __init__(self, data: TableTensor):
        """
        Initialize the executor with a TableTensor.

        Args:
            data (TableTensor): The tensor-based representation of a relational table.
        """
        self.data = data
        self.compiler = KeroCompiler(data)

    def execute(self, kquery: Dict[str, Any]) -> torch.Tensor:
        """
        Execute a kquery on the initialized data using tensor operations.

        Args:
            kquery (Dict[str, Any]): Intermediate representation of the query.

        Returns:
            torch.Tensor: Final data after executing all operations.
        """
        # Compile and execute operations from kquery
        result = self.compiler.compile(kquery['operations'])
        
        # If kquery specifies columns, select those columns from the result
        if 'columns' in kquery:
            selected_columns = kquery['columns']
            result = self._select_columns(result, selected_columns)
        
        return result

    def _select_columns(self, result: torch.Tensor, columns: List[str]) -> torch.Tensor:
        """
        Select specific columns from the result tensor.

        Args:
            result (torch.Tensor): Resulting tensor after executing operations.
            columns (List[str]): List of column names to select.

        Returns:
            torch.Tensor: Tensor with selected columns.
        """
        # Assuming result is a tensor where each column corresponds to a field in the original data
        # This method needs adjustment based on how columns are indexed in the result tensor
        # For simplicity, let's assume columns are directly indexable
        column_indices = [list(self.data.columns.keys()).index(col) for col in columns]
        return result[:, column_indices]
