from kero._engine._kero.dialects import func, db, arith
from kero._engine._kero.passmanager import PassManager
from kero.engine import _keroEngine
import kero._engine._kero.ir as ir

from .parser.dbast import *

import typing as t
import pyarrow as pa


PYARROW_TO_DB_TYPES: t.Dict[pa.DataType, str] = {
    # Integer Types
    pa.int8(): "i8",
    pa.int16(): "i16",
    pa.int32(): "i32",
    pa.int64(): "i64",

    # Floating Types
    pa.float16(): "f16",
    pa.float32(): "f32",
    pa.float64(): "f64",
}

def make_dbtable_type(node: DBTable, context):
    table_metadata = node.metadata.metadata

    num_rows = table_metadata.get("num_rows", -1)
    num_cols = table_metadata.get("num_cols", 0)

    column_attr_t = []
    for column_name in table_metadata.get("column_names", []):
        column_attr_t.append(
            f'#db.column<"{column_name}", {convert_dtype(node.get_column_dtype(column_name))}, {num_rows}>'
        )

    table_t = f'!db.table<{num_cols}, {num_rows} : [{", ".join(column_attr_t)}]>'
    return ir.Type.parse(table_t, context)

def make_dbcolumn_type(dtype, context):
    if dtype in PYARROW_TO_DB_TYPES:
        dtype = convert_dtype(dtype)

    return ir.Type.parse(f"!db.column<{dtype}>", context)

def convert_dtype(dtype):
    return PYARROW_TO_DB_TYPES[dtype]


class IRGen:
    def __init__(self, query_name: str, operations):
        self.operations = operations
        self.context = ir.Context()
        self.loc = ir.Location.unknown(self.context)
        self.module = ir.Module.create(self.loc)
        self.func_name = query_name
        self.generator = None
        _keroEngine.register_dialect(self.context)

    def emit_ir(self):
        with self.context:
            with self.loc, ir.InsertionPoint(self.module.body):
                first_op = self.operations[0]
                last_op = self.operations[-1]

                in_table_t = make_dbtable_type(first_op.input, self.context)
                out_table_t = make_dbtable_type(last_op.output, self.context)

                ftype = ir.FunctionType.get(inputs=[in_table_t], results=[out_table_t])
                func_op = func.FuncOp(self.func_name, ftype)
                func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                func_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
                self.generator = AstToKeroConverter(self.context, self.loc, self.module, func_block)

                out = func_block.arguments[0]
                for operation in self.operations:
                    out = self.generator.resolve_node(operation, out)

                with ir.InsertionPoint(func_block):
                    func.return_([out])


class AstToKeroConverter:
    def __init__(
        self,
        context: ir.Context,
        location: ir.Location,
        module: ir.Module,
        block: ir.Block,
    ):
        self.block = block
        self.context = context
        self.loc = location
        self.module = module
        self.op_map = self._node_to_op_map()

    def resolve_node(self, node, inputs):
        with self.loc, ir.InsertionPoint(self.block):
            return self.op_map[type(node)](node, inputs)

    def _make_scan_op(self, node: ScanOp, inputs):
        table_t = make_dbtable_type(node.input, self.context)
        return db.scan(output=table_t, table=inputs)

    def _make_output_op(self, node: OutputOp, inputs):
        table_t = make_dbtable_type(node.output, self.context)
        return db.output(output=table_t, table=inputs, select=node.select)

    def _make_filter_op(self, node: FilterOp, inputs):
        output_table_t = make_dbtable_type(node.output, self.context)

        filter_op = db.FilterOp(filtered=output_table_t, table=inputs)

        col_types = []
        col_names = []
        col_dtypes = []
        for col in node.input.columns:
            dtype = col.metadata.metadata["column_dtype"]
            col_types.append(make_dbcolumn_type(dtype, self.context))
            col_names.append(col.metadata.metadata["column_name"])
            col_dtypes.append(convert_dtype(dtype))

        filter_block = ir.Block.create_at_start(filter_op.body, col_types)

        with ir.InsertionPoint(filter_block):
            cmp_op = node.region.operations
            col_idx = col_names.index(cmp_op.lhs.name)
            col_arg = filter_block.arguments[col_idx]

            bitwidth = int(col_dtypes[col_idx][1:])
            int_type = ir.IntegerType.get(bitwidth)
            const = arith.constant(int_type, cmp_op.rhs.constant)

            predicate_map = {"lt": 0, "lte": 1, "gt": 2, "gte": 3, "eq": 4, "neq": 5}
            predicate_attr = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64, context=self.context),
                predicate_map[cmp_op.predicate],
            )

            result_type = make_dbcolumn_type("i1", self.context)
            cmp_result = db.cmpi(
                result=result_type,
                predicate=predicate_attr,
                col=col_arg,
                constint=const,
            )

            db.filter_yield(mask=cmp_result)

        return filter_op.filtered

    def _node_to_op_map(self):
        return {
            ScanOp: self._make_scan_op,
            OutputOp: self._make_output_op,
            FilterOp: self._make_filter_op,
        }


def db_to_llvm_lowering(module, context):
    with context:
        pm = PassManager.parse(
            "builtin.module("
            "db-to-tensor-and-linalg,"
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
        pm.run(module.operation)

        return module, context

def db_to_tensor(module, context):
    with context:
        pm = PassManager.parse("builtin.module(db-to-tensor-and-linalg)")
        pm.run(module.operation)
        return module, context
