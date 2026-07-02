from unittest import TestCase

class TestParser(TestCase):
    def test_simple_parsing(self):
        # Parser Testing Stale for phase1
        from kero.arrow.samples import toy_school_dataset
        from kero.engine.parser import Parser
        
        dataset = toy_school_dataset()
        parser = Parser(dataset)
        query = "SELECT salary, age from person WHERE age > 10"
        out = parser.parse(query)
