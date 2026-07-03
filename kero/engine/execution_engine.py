import kero._engine._kero.execution_engine as execute
import kero._engine._kero.runtime.np_to_memref as runtime
from kero.arrow.data import Dataset

import ctypes

import numpy as np

class KeroEngine:
    def __init__(self, module, context, opt_level=3):
        self.module = module
        self.context = context
        with self.context:
            self.executor = execute.ExecutionEngine(module, opt_level=opt_level)

    def execute(self, query_name, data: Dataset, inputs, outputs):
        # input: List of Table names
        # outputs: List of output arrays needs to be created
        with self.context:
            pointer_inputs = self.get_input_from_data(data, inputs)
            pointer_outputs = self.get_pointers_from_output(outputs)

            self.executor.invoke(query_name, *pointer_inputs, *pointer_outputs)

    def get_input_from_data(self, data: Dataset, inputs):
        pointer_inputs = []
        arrays = []

        for i in inputs:
            arrays.extend(data.get_table_as_arrays(i))
            
        for array in arrays:
            mem = runtime.get_ranked_memref_descriptor(array)
            pointer_inputs.append(ctypes.pointer(mem))

        return pointer_inputs

    def get_pointers_from_output(self, outputs):
        pointer_outputs = []

        for output in outputs:
            mem = runtime.get_ranked_memref_descriptor(np.empty(0, dtype=output.dtype))
            pointer_outputs.append(ctypes.pointer(mem))

        return pointer_outputs
