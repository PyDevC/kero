import torch

_operations = { # example operators
    "sum": torch.add,
    "mean": lambda x, y: (x + y) / 2,
    "median": lambda x, y: torch.median(torch.stack([x, y]))
}

class Operator: # example class. Will be implemented in detail later
    def __init__(self, name: str, tensor1: torch.Tensor, tensor2: torch.Tensor):
        if name not in _operations:
            raise ValueError(f"Unsupported operation: {name}")
        self.name = name
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def execute(self) -> torch.Tensor:
        op_func = _operations[self.name]
        return op_func(self.tensor1, self.tensor2)

class DiGraph:
    def __init__(self, operators: list[str], operands: list[torch.Tensor]):
        if len(operands) != 2:
            raise ValueError("Invalid number of operators or operands.")
        
        self.operators = operators
        self.operands = operands
        self.nodes = []  # will store Operator instances

    def _build_graph(self):
        operand_queue = self.operands.copy()
        # here we are assuming that all the operands tensors are in same order they need to executed 
        # If this doesn't work then we will look into it later on
        for op in self.operators:
            t1 = operand_queue.pop(0)
            t2 = operand_queue.pop(0)
            operator_node = Operator(op, t1, t2)
            result = operator_node.execute()
            operand_queue.insert(0, result)  # result replaces operands
            self.nodes.append(operator_node)

        self.result = operand_queue[0]  # Final result after all operations

    def execute(self) -> torch.Tensor:
        self._build_graph()
        return self.result
