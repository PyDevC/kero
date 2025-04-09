import torch
from abc import ABC
from .base import operator

class logical_op(ABC, operator):
    def __init__(self, tensor1, tensor2):
        super().__init__()
        self.operand1 = tensor1
        self.operand2 = tensor2
        self._check_type()

    def _check_type(self):
        if isinstance(self.operand2, int):
            self.operand2 = torch.tensor(self.operand2)

class eq(logical_op):
    def execute(self):
        return (self.operand1 == self.operand2).bool()


class ne(logical_op):
    def execute(self):
        return (self.operand1 != self.operand2).bool()


class gt(logical_op):
    def execute(self):
        return torch.greater(self.operand1, self.operand2)


class ge(logical_op):
    def execute(self):
        return torch.greater_equal(self.operand1, self.operand2)


class lt(logical_op):
    def execute(self):
        return torch.less(self.operand1, self.operand2)


class le(logical_op):
    def execute(self):
        return torch.less_equal(self.operand1, self.operand2)
