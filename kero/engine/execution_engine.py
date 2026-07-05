import kero._engine._kero.execution_engine as execute
import kero._engine._kero.runtime.np_to_memref as runtime
from kero.arrow.data import Dataset

import ctypes

import numpy as np


class Memref1D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_longlong),
        ("aligned", ctypes.POINTER(ctypes.c_int32)),
        ("offset", ctypes.c_longlong),
        ("shape", ctypes.c_longlong * 1),
        ("strides", ctypes.c_longlong * 1),
    ]


class OutputStruct(ctypes.Structure):
    _fields_ = [
        ("memref0", Memref1D),
        ("memref1", Memref1D),
        ("memref2", Memref1D),
    ]


class KeroEngine:
    def __init__(self, module, context, opt_level=3):
        self.module = module
        self.context = context
        with self.context:
            self.executor = execute.ExecutionEngine(module, opt_level=opt_level)

    def execute(self, query_name, data: Dataset, inputs, outputs):
        with self.context:
            pointer_inputs = self.get_input_from_data(data, inputs)

            out_struct = OutputStruct()
            out_ptr = ctypes.pointer(out_struct)

            self.executor.invoke(
                query_name,
                ctypes.byref(out_ptr),
                *[ctypes.byref(p) for p in pointer_inputs])

            return [ctypes.pointer(out_struct.memref0),
                    ctypes.pointer(out_struct.memref1),
                    ctypes.pointer(out_struct.memref2)]

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
