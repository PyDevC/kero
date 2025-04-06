import torch
from abc import ABC
from .base import operator

class agg_op(ABC, operator):
    def __init__(self, optype, tensor):
        super().__init__(optype)
        self.operand = tensor


class sum(agg_op):
    def execute(self):
        return torch.sum(self.operand)


class mean(agg_op):
    def execute(self):
        return torch.mean(self.operand)


class median(agg_op):
    def execute(self):
        return torch.median(self.operand)


class max(agg_op):
    def execute(self):
        return torch.max(self.operand)


class min(agg_op):
    def execute(self):
        return torch.min(self.operand)


class std(agg_op):
    def execute(self):
        return torch.std(self.operand)


class var(agg_op):
    def execute(self):
        return torch.var(self.operand)


class argmax(agg_op):
    def execute(self):
        return torch.argmax(self.operand)


class argmin(agg_op):
    def execute(self):
        return torch.argmin(self.operand)
