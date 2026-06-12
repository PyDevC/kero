import kero.engine.parser
from unittest import TestCase

class TestParser(TestCase):
    def test_simple_parsing(self):
        from kero.arrow.samples import toy_school_dataset
        from kero.engine.parser import Parser
        
        dataset = toy_school_dataset()
        
        parser = Parser(dataset)
        
        query = "SELECT age from person"
        
        out = parser.parse(query)
        assert out
        assert len(out) == 2
