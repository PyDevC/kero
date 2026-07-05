"""Exceptions for Arrow utils"""

class ArrowUtilsException(Exception): ...

class TableNotFoundException(ArrowUtilsException):
    def __init__(self, message):
        super().__init__(message)

class NodeTypeResolveException(ArrowUtilsException):
    def __init__(self, message):
        super().__init__(message)
