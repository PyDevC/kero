import torch
from abc import ABC
from .base import operator

class bi_mat_op(ABC, operator):
    def __init__(self, optype, tensor1, tensor2):
        super().__init__(optype)
        self.operand1 = tensor1
        self.operand2 = tensor2

class u_math_op(ABC, operator):
    def __init__(self, optype: str, tensor: torch.Tensor):
        super().__init__(optype)
        self.operand = tensor

class add(bi_mat_op):
    def execute(self) -> torch.Tensor:
        return torch.add(self.operand1, self.operand2)

class sub(bi_mat_op):
    def execute(self) -> torch.Tensor:
        return torch.sub(self.operand1, self.operand2)

class prod(bi_mat_op):
    def execute(self) -> torch.Tensor:
        return torch.prod(self.operand1, self.operand2)

class div(bi_mat_op):
    def execute(self) -> torch.Tensor:
        return torch.div(self.operand1, self.operand2)

class pow(bi_mat_op):
    def execute(self) -> torch.Tensor:
        return torch.pow(self.operand1, self.operand2)


class exp(u_math_op):
    def execute(self):
        return torch.exp(self.operand)


class log(u_math_op):
    def execute(self):
        return torch.log(self.operand)


class sqrt(u_math_op):
    def execute(self):
        return torch.sqrt(self.operand)


class abs(u_math_op):
    def execute(self):
        return torch.abs(self.operand)


class ceil(u_math_op):
    def execute(self):
        return torch.ceil(self.operand)


class floor(u_math_op):
    def execute(self):
        return torch.floor(self.operand)


class round(u_math_op):
    def execute(self):
        return torch.round(self.operand)
