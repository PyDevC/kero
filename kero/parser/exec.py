class BaseParserError(Exception):
    """Base Class for all parser exception"""
    pass

class NodeNotFound(BaseParserError):
    pass
