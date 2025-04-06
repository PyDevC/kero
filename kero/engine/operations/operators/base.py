import torch
from abc import ABC, abstractmethod

class operator():
    def __init__(self, optype):
        self.opname = self.__class__.__name__
        self.optype = optype

    @abstractmethod
    def execute(self) -> torch.Tensor:
        pass
