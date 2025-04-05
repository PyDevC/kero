import torch

# Convert SQL data into tensor as:
# TensorTable, NumericalColumnTensor, StringTensor, ScalerTensor, DateTensor, etc.

# The below are classes in which each tensor will be represented

class TableTensor: 
    def __init__(self,name: str, tensor: torch.Tensor, attributes: tuple[str], dim=None):
        """Tensor Table is representation of Relational table in form of tensors
        it only contains table name and attributes. Each table attribute points 
        to the column tensor

        Parameters:
            tensor: converting a pre-existing tensor to tensor table
            attributes: contains column names of the table
            dim: size of the tensor
        """
        # TODO: assign name to table
        # give a choice to whether pass a tensor as table or pass attributes
        self.tensor = tensor
        self.dim = dim
        self.attributes = attributes


class NumTensor:
    def __init__(self,column_name: str, tensor: torch.Tensor):
        """Number Tensor is the tensor representation of a numerical column of 
        Relational database. Ex: age, size, etc. columns
        The NumTensor has to be of size (n, 1)
        """
        n = len(tensor)
        self.tensor = tensor
        self.dim = (n,1)
        if tensor.shape != self.dim:
            raise TypeError # need to change errors

class StrTensor:
    def __init__(self, column_name: str, tensor:torch.Tensor):
        """String Tensor is the representation of a categorical column of 
        Relational database. Ex: name, id, etc columns.
        The size of String Tensor has to be of (n, m) where n is number of column
        and m is the size of the largest string in the column
        """
        n = len(tensor)
        # m = len(tensor.) # its better to get the size of longest string at the time of retrival
        self.tensor = tensor
        self.dim = (n, m)

        if tensor.dtype != torch.CharTensor:
            raise


class ScalerTensor:
    def __init__(self, scaler: torch.Tensor):
        """Scaler Tensors are 0-D tensors that represents the constants in the
        Relational database.
        Strict type checking of dimension of Scaler Tensor
        """
        self.tensor = scaler
        self.dim = (1,)

class DateTensor:
    def __init__(self, column_name: str, tensor, format="DD-MM-YYYY"):
        """Date Tensor are representation of Date column in Relational database
        This is stored in a unique format. 
        
        Each row entry has to be in same format as mentioned
        the dimension of the tensor is (n, 3) where represents the date
        """
        n = len(tensor)
        self.tensor = tensor
        self.dim = (n, 3)
