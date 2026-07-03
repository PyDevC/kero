from unittest import TestCase
import kero._engine._kero.ir as ir
from kero.engine import _keroEngine
from kero._engine._kero.dialects import func, db
import ctypes
import numpy as np
from unittest import TestCase
import kero._engine._kero.ir as ir
from kero._engine._kero.execution_engine import ExecutionEngine
from kero._engine._kero.runtime import get_ranked_memref_descriptor, ranked_memref_to_numpy
from kero._engine._kero.passmanager import PassManager

def mlir_source():
    return """
#map = affine_map<(d0) -> (d0)>
module {
  func.func @main(%arg0: tensor<100xi32>, %arg1: tensor<100xi32>, %arg2: tensor<100xi32>) -> (tensor<?xi32>) attributes { llvm.emit_c_interface } {
    %0 = tensor.empty() : tensor<100xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<100xi32>, tensor<100xi32>, tensor<100xi32>) outs(%0 : tensor<100xi1>) {
    ^bb0(%in: i32, %in_1: i32, %in_2: i32, %out: i1):
      %c10_i32 = arith.constant 10 : i32
      %8 = arith.cmpi eq, %in_1, %c10_i32 : i32
      linalg.yield %8 : i1
    } -> tensor<100xi1>
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %2 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %c0_0) -> (index) {
      %extracted = tensor.extract %1[%arg3] : tensor<100xi1>
      %8 = scf.if %extracted -> (index) {
        %9 = arith.addi %c1, %arg4 : index
        scf.yield %9 : index
      } else {
        scf.yield %arg4 : index
      }
      scf.yield %8 : index
    }
    %3 = tensor.empty(%2) : tensor<?xi32>
    %4 = tensor.empty(%2) : tensor<?xi32>
    %5 = tensor.empty(%2) : tensor<?xi32>
    %6:4 = scf.for %arg3 = %c0 to %c100 step %c1 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %5, %arg7 = %c0_0) -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index) {
      %extracted = tensor.extract %1[%arg3] : tensor<100xi1>
      %7:4 = scf.if %extracted -> (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index) {
        %extracted_1 = tensor.extract %arg0[%arg3] : tensor<100xi32>
        %inserted = tensor.insert %extracted_1 into %arg4[%arg7] : tensor<?xi32>
        %extracted_2 = tensor.extract %arg1[%arg3] : tensor<100xi32>
        %inserted_3 = tensor.insert %extracted_2 into %arg5[%arg7] : tensor<?xi32>
        %extracted_4 = tensor.extract %arg2[%arg3] : tensor<100xi32>
        %inserted_5 = tensor.insert %extracted_4 into %arg6[%arg7] : tensor<?xi32>
        %8 = arith.addi %c1, %arg7 : index
        scf.yield %inserted, %inserted_3, %inserted_5, %8 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
      } else {
        scf.yield %arg4, %arg5, %arg6, %arg7 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
      }
      scf.yield %7#0, %7#1, %7#2, %7#3 : tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, index
    }
    return %6#0 : tensor<?xi32>
  }
}
    """

class TestExecution(TestCase):
    def setUp(self):
        self.context = ir.Context()
        self.location = ir.Location.unknown(self.context)
        self.module = None
        _keroEngine.register_dialect(self.context)

        self.source = mlir_source()

    def test_source_execution(self):
        with self.context, self.location:
            self.module = ir.Module.parse(self.source)
            pm = PassManager.parse(
                "builtin.module("
                "one-shot-bufferize{bufferize-function-boundaries=true},"
                "convert-linalg-to-loops,"
                "expand-strided-metadata,"
                "convert-scf-to-cf,"
                "convert-to-llvm,"
                "finalize-memref-to-llvm,"
                "convert-func-to-llvm,"
                "convert-arith-to-llvm"
                ")"
            )
            pm.run(self.module.operation)

            exe = ExecutionEngine(self.module, opt_level=3)

            age_in = np.random.randint(1, 90, size=100, dtype=np.int32)
            salary_in = np.array([10 if i % 10 == 0 else 50 for i in range(100)], dtype=np.int32)
            budget_in = np.random.randint(100, 1000, size=100, dtype=np.int32)
            
            out_salary = np.empty(0, dtype=np.int32)
            
            arg0 = ctypes.pointer(get_ranked_memref_descriptor(age_in))
            arg1 = ctypes.pointer(get_ranked_memref_descriptor(salary_in))
            arg2 = ctypes.pointer(get_ranked_memref_descriptor(budget_in))
            res0 = ctypes.pointer(get_ranked_memref_descriptor(out_salary))
            
            exe.invoke("main", 
                       ctypes.byref(res0), 
                       ctypes.byref(arg0), 
                       ctypes.byref(arg1), 
                       ctypes.byref(arg2)
            )
            
            result = ranked_memref_to_numpy(res0)
            assert(result.size == 10)
