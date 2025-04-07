import torch

from typing import Dict, List, Union

class BaseTensor:

    def __init__(self, tensor: torch.Tensor, name: str):
        self.name = name
        self.tensor = tensor
        self._validate_tensor()

    def _validate_tensor(self):
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError (
                f"Expected torch.Tensor, but got {type(self.tensor)} instead."
            )


class NumTensor(BaseTensor):
    """Number Tensor is the tensor representation of a numerical column of 
    Relational database. Ex: age, size, etc. columns
    The NumTensor has to be of size (n, 1)
    """

    def _validate_tensor(self):
        super()._validate_tensor()


class StrTensor(BaseTensor):
    """String Tensor is the representation of a categorical column of 
    Relational database. Ex: name, id, etc columns.
    The size of String Tensor has to be of (n, m) where n is number of column
    and m is the size of the largest string in the column
    """

    def __init__(self, data: List[str], name: str):
        self.vocab = {word: idx for idx, word in enumerate(set(data))}
        tensor = torch.tensor([self.vocab[word] for word in data], dtype=torch.long)
        super().__init__(tensor, name)


class ScalerTensor:
    """Scaler Tensors are 0-D tensors that represents the constants in the
    Relational database.
    Strict type checking of dimension of Scaler Tensor
    """

    def __init__(self, scaler: str, name: str):
        self.name = name
        self.tensor = torch.tensor(scaler)


class DateTensor(BaseTensor):

    def __init__(self, dates: List[str], name: str):
        """Date Tensor are representation of Date column in Relational database
        This is stored in a unique format. 
        
        Each row entry has to be in same format as mentioned
        the dimension of the tensor is (n, 3) where represents the date
        """
        tensor = torch.tensor([
            self._date_to_ordinal(date_str) for date_str in dates
        ])
        
        super().__init__(tensor, name)


    def _date_to_ordinal(self, date):
        return date


class TableTensor: 

    def __init__(
        self, 
        columns: Dict[str, Union[BaseTensor, NumTensor, StrTensor, DateTensor]], 
        name: str
    ):
        """Tensor Table is representation of Relational table in form of tensors
        it only contains table name and attributes. Each table attribute points 
        to the column tensor

        Parameters:
            tensor: converting a pre-existing tensor to tensor table
            name: name of the table
        """
        self.name = name
        self.columns = columns
        self._validate_columns()


    def _validate_columns(self):
        self._validate_dtype()
        self._validate_shape()


    def _validate_dtype(self):
        pass


    def _validate_shape(self):
        if not self.columns:
            return 

        first_col = next(iter(self.columns.values()))
        num_rows = len(first_col.tensor)

        for name, tensor in self.columns.items():
            if len(tensor.tensor) != num_rows:
                raise ValueError(
                    f"Column {name} has {len(tensor.tensor)} rows, "
                     "expected {num_rows}"
                )

    def to(self, device: str):
        # do not run
        return TableTensor(
            {name: tensor.tensor.to(device) for name, tensor in self.columns.items()},
            self.name
        )
