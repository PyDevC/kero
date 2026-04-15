from .. import exec

KeroException = exec.KeroError

class ParserError(KeroException):
    """Raised due to incorrect parser handling."""
    def __init__(self, message: str, status_code=None):
        super().__init__(message)
        self.status_code = status_code

# Parsing errors
class NodeNotFoundError(ParserError):
    """Raised when a particular Node does not exists in AST."""
    pass
