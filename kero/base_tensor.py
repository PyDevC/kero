import torch

# Convert SQL data into tensor as:
# TensorTable, NumericalColumnTensor, StringTensor, ScalerTensor, DateTensor, etc.

# The below are classes in which each tensor will be represented

class TableTensor: 
    def __init__(self, tensor: torch.Tensor, name: str, attributes: list[str], dim=None):
        """Tensor Table is representation of Relational table in form of tensors
        it only contains table name and attributes. Each table attribute points 
        to the column tensor

        Parameters:
            tensor: converting a pre-existing tensor to tensor table
            name: name of the table
            attributes: contains column names of the table
            dim: size of the tensor
        """
        self.name = name
        self.tensor = tensor
        self.dim = dim
        self.attributes = attributes

    def add_attribute(self, column): # change the type hint to column: pointer to tensor
        """Add a new attribute to the table is the query results a new column
        """
        self.attributes.append(column)


class NumTensor:
    def __init__(self, tensor: torch.Tensor, name: str):
        """Number Tensor is the tensor representation of a numerical column of 
        Relational database. Ex: age, size, etc. columns
        The NumTensor has to be of size (n, 1)
        """
        self.name = name
        self.tensor = tensor
        self.dim = tensor.size()
        if len(self.dim) != 2 and self.dim[-1] == 1:
            raise ValueError(
                "Invalid tensor shape: The shape must be of (n,1)"
            )

class StrTensor:
    def __init__(self, tensor:torch.Tensor, name: str):
        """String Tensor is the representation of a categorical column of 
        Relational database. Ex: name, id, etc columns.
        The size of String Tensor has to be of (n, m) where n is number of column
        and m is the size of the largest string in the column
        """
        self.name = name
        self.tensor = tensor
        self.dim = tensor.size()
        if len(self.dim) != 2:
            raise ValueError(
                "Invalid tensor shape: The shape must be of (n,m)"
            )


        if tensor.dtype != torch.CharTensor:
            raise TypeError(
                f"""Invalid tensor type, attempting to assign {tensor.dtype} to StrTensor
                tensor must be of type{type(torch.CharTensor)}
                """
            )


class ScalerTensor:
    def __init__(self, scaler: int, name: str):
        """Scaler Tensors are 0-D tensors that represents the constants in the
        Relational database.
        Strict type checking of dimension of Scaler Tensor
        """
        self.name = name
        self.tensor = torch.tensor(scaler)
        self.dim = 0

class DateTensor:
    def __init__(self, tensor: torch.Tensor, name:str, format="DD-MM-YYYY"):
        """Date Tensor are representation of Date column in Relational database
        This is stored in a unique format. 
        
        Each row entry has to be in same format as mentioned
        the dimension of the tensor is (n, 3) where represents the date
        """
        self.name = name
        self.tensor = tensor
        self.dim = tensor.size()
        self.format = format
        if len(self.dim) != 2 and self.dim[-1] == 3:
            raise ValueError(
                "Invalid tensor shape: The shape must be of (n,1)"
            )
    def _check_date_format(self):
        """checks if all the dates are in same format as self.format
        """
        pass
