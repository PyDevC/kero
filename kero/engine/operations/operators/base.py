import torch
from abc import ABC, abstractmethod

class operator():
    def __init__(self):
        self.opname = self.__class__.__name__
