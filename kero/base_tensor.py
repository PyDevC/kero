import torch

# Convert SQL data into tensor as:
# TensorTable, NumericalColumnTensor, StringTensor, ScalerTensor, DateTensor, etc.

# The below are classes in which each tensor will be represented

class TensorTable: # need some time think about its development
    def __init__(self):
        self.tensor = None
        self.dim = ()

class NumTensor:
    def __init__(self):
        self.tensor = None
        self.dim = ()

class StrTensor:
    def __init__(self):
        self.tensor = None 
        self.dim = ()

class ScalerTensor:
    def __init__(self):
        self.tensor = None 
        self.dim = ()

class DateTensor:
    def __init__(self):
        self.tensor = None 
        self.dim = ()
