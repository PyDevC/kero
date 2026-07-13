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


def create_dynamic_struct_from_context(execution_context):
    sorted_ids = sorted(execution_context.memref_structs.keys())
    class DynamicOutputStruct(ctypes.Structure):
        _fields_ = [(f'memref{i}', Memref1D) for i in sorted_ids]

    return DynamicOutputStruct()

class ExecutionContext:
    def __init__(self):
        self.results = {}
        self.tuple_counts = {}
        self.memref_structs = {}

    def add_result_slot(self, result_id: int):
        self.memref_structs[result_id] = Memref1D()
        self.tuple_counts[result_id] = 0

    def get_memref_struct(self, result_id: int):
        return self.memref_structs.get(result_id)

    def set_tuple_count(self, result_id: int, count: int):
        self.tuple_counts[result_id] = count

    def get_tuple_count(self, result_id: int):
        return self.tuple_counts.get(result_id, 0)

    def extract_size_from_memref(self, memref_ptr):
        memref = memref_ptr.contents
        if hasattr(memref, 'shape'):
            return memref.shape[0]

        return 0

    def update_sizes_from_output(self, output_struct):
        for result_id in self.memref_structs.keys():
            memref = getattr(output_struct, f'memref{result_id}')
            size = self.extract_size_from_memref(memref)
            self.set_tuple_count(result_id, size)


class KeroEngine:
    def __init__(self, module, context, opt_level=3):
        self.module = module
        self.context = context
        with self.context:
            self.executor = execute.ExecutionEngine(module, opt_level=opt_level)

        self.execution_context = ExecutionContext()

    def configure_outputs(self, result_ids):  
        for result_id in result_ids:  
            self.execution_context.add_result_slot(result_id)  

    def init_memref(self):
        out_struct = create_dynamic_struct_from_context(self.execution_context)
        for result_id in self.execution_context.memref_structs.keys():
            memref = getattr(out_struct, f"memref{result_id}")
            buffer_size = 1024  # Initial buffer size 1 KB
            memref.allocated = buffer_size
            memref.aligned = (ctypes.c_int32 * buffer_size)()
            memref.offset = 0
            memref.shape[0] = 0
            memref.strides[0] = 1

        return out_struct

    def execute(self, query_name, data: Dataset, inputs):
        with self.context:
            pointer_inputs = self.get_input_from_data(data, inputs)

            out_struct = self.init_memref()
            out_ptr = ctypes.pointer(out_struct)

            self.executor.invoke(
                query_name,
                ctypes.byref(out_ptr),
                *[ctypes.byref(p) for p in pointer_inputs])
              
            results = {}
            for result_id in self.execution_context.memref_structs.keys():
                memref = getattr(out_struct, f"memref{result_id}")
                results[result_id] = ctypes.pointer(memref)
                self.execution_context.set_tuple_count(result_id, memref.shape[0])

            return results


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

    def results_to_numpy(self, results):
        numpy_results = {}
        for result_id, memref in results.items():
            size = self.execution_context.get_tuple_count(result_id)
            array = runtime.ranked_memref_to_numpy(memref)
            numpy_results[result_id] = array
        return numpy_results
