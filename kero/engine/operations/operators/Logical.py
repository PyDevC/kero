import torch
from abc import ABC
from .base import operator

class logical_op(ABC, operator):
    def __init__(self, optype, tensor1, tensor2):
        super().__init__(optype)
        self.operand1 = tensor1
        self.operand2 = tensor2

class eq(logical_op):
    def execute(self):
        return torch.eq(self.operand1, self.operand2)


class ne(logical_op):
    def execute(self):
        return torch.ne(self.operand1, self.operand2)


class gt(logical_op):
    def execute(self):
        return torch.gt(self.operand1, self.operand2)


class ge(logical_op):
    def execute(self):
        return torch.ge(self.operand1, self.operand2)


class lt(logical_op):
    def execute(self):
        return torch.lt(self.operand1, self.operand2)


class le(logical_op):
    def execute(self):
        return torch.le(self.operand1, self.operand2)
