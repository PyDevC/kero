from kero.engine import _keroEngine
from kero._engine._kero.dialects import func, db, arith
import kero._engine._kero.ir as ir
from kero._engine._kero.passmanager import PassManager

from .parser.dbast import ScanOp, OutputOp, FilterOp


def make_dbtable_type(node, context, dynamic_rows=False):
    ncols = len(node.columns)
    if ncols == 0:
        raise ValueError("Cannot create table type with 0 columns")

    nrows_table = node.metadata.metadata.get("nrows", 0)
    if nrows_table == 0 and ncols > 0:
        nrows_table = node.columns[0].metadata.metadata.get("nrows", 0)

    cols = []
    for col in node.columns:
        name = col.metadata.metadata["name"]
        dtype = col.metadata.metadata["dtype"]
        n = -1 if dynamic_rows else col.metadata.metadata["nrows"]
        cols.append(f'#db.column<"{name}", {dtype}, {n}>')
    nrows_str = -1 if dynamic_rows else nrows_table
    type_str = f'!db.table<{ncols}, {nrows_str} : [{", ".join(cols)}]>'
    return ir.Type.parse(type_str, context)


def make_dbcolumn_type(dtype_str, context):
    return ir.Type.parse(f"!db.column<{dtype_str}>", context)


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
                first = self.operations[0]
                last = self.operations[-1]

                if isinstance(first, ScanOp):
                    in_table_t = make_dbtable_type(first.table, self.context)
                else:
                    in_table_t = make_dbtable_type(first.input, self.context)

                has_filter = any(isinstance(op, FilterOp) for op in self.operations)
                if isinstance(last, FilterOp):
                    out_table_t = make_dbtable_type(last.output, self.context, dynamic_rows=True)
                elif isinstance(last, OutputOp):
                    out_table_t = make_dbtable_type(last.output, self.context, dynamic_rows=has_filter)
                else:
                    out_table_t = make_dbtable_type(last.table, self.context)

                ftype = ir.FunctionType.get(inputs=[in_table_t], results=[out_table_t])
                func_op = func.FuncOp(self.func_name, ftype)
                func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
                self.generator = AstToKeroConverter(
                    self.context, self.loc, self.module, entry_block, has_filter=has_filter
                )

                out = entry_block.arguments[0]
                for operation in self.operations:
                    out = self.generator.resolve_node(operation, out)

                with ir.InsertionPoint(entry_block):
                    func.return_([out])


class AstToKeroConverter:
    def __init__(
        self,
        context: ir.Context,
        location: ir.Location,
        module: ir.Module,
        block: ir.Block,
        has_filter=False,
    ):
        self.block = block
        self.context = context
        self.has_filter = has_filter
        self.loc = location
        self.module = module
        self.op_map = self._node_to_op_map()

    def resolve_node(self, node, inputs):
        with self.loc, ir.InsertionPoint(self.block):
            return self.op_map[type(node)](node, inputs)

    def _make_scan_op(self, node: ScanOp, inputs):
        table_t = make_dbtable_type(node.table, self.context)
        return db.scan(output=table_t, table=inputs)

    def _make_output_op(self, node: OutputOp, inputs):
        table_t = make_dbtable_type(node.output, self.context, dynamic_rows=self.has_filter)
        select = [col.metadata.metadata["name"] for col in node.output.columns]
        return db.output(output=table_t, table=inputs, select=select)

    def _make_filter_op(self, node: FilterOp, inputs):
        input_table_t = make_dbtable_type(node.input, self.context)
        output_table_t = make_dbtable_type(node.output, self.context, dynamic_rows=True)

        filter_op = db.FilterOp(filtered=output_table_t, table=inputs)

        col_types = []
        col_names = []
        col_dtypes = []
        for col in node.input.columns:
            dtype_str = col.metadata.metadata["dtype"]
            col_types.append(make_dbcolumn_type(dtype_str, self.context))
            col_names.append(col.metadata.metadata["name"])
            col_dtypes.append(dtype_str)

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
