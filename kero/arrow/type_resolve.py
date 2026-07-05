import typing as t

import pyarrow as pa

from kero.arrow.data import Dataset
from kero.engine.parser import dbast
from kero.arrow.exce import NodeTypeResolveException

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

def resolve_node(dataset: Dataset, node):
    resolve_func = NODE_RESOLVE_MAP.get(type(node))
    if resolve_func is not None:
        return resolve_func(dataset, node)

    raise NodeTypeResolveException(
        f"node can't be resolved, unknown node type: {type(node)}"
    )

         
def resolve_dbtable(dataset: Dataset, node: dbast.DBTable):
    table_name = node.metadata.metadata["name"]
    table = dataset.get_table(table_name)
    node.metadata.metadata["ncols"] = table.num_columns
    node.metadata.metadata["nrows"] = table.num_rows

    schema = table.schema
    columnattr = node.columns
    if columnattr[0].metadata.metadata["is_star"]:
        new_columnattr = []
        node.columns.clear()
        for field in schema:
            metadata = dbast.Metadata()
            metadata.metadata["name"] = field.name
            metadata.metadata["dtype"] = PYARROW_TO_DB_TYPES[field.type]
            metadata.metadata["nrows"] = table.num_rows
            new_columnattr.append(dbast.DBColumnAttr(metadata))

        node.columns = new_columnattr
        return

    for column in node.columns:
        col_name = column.metadata.metadata["name"]
        field = schema.field(col_name)
        column.metadata.metadata["dtype"] = PYARROW_TO_DB_TYPES[field.type]
        column.metadata.metadata["nrows"] = table.num_rows


def resolve_dbcolumn(dataset: Dataset, node: dbast.DBColumn):
    if not node.name:
        return
    for table_name in dataset.tables:
        table = dataset.tables[table_name]
        schema = table.schema
        if node.name in schema.names:
            field = schema.field(node.name)
            node.dtype = PYARROW_TO_DB_TYPES[field.type]
            return


def resolve_dbscan_op(dataset: Dataset, node: dbast.ScanOp):
    resolve_dbtable(dataset, node.table)


def resolve_dbfilter_op(dataset: Dataset, node: dbast.FilterOp):
    resolve_dbcmpi_op(dataset, node.region.operations)
    resolve_dbcolumn(dataset, node.region.f_yield.mask)


def resolve_dboutput_op(dataset: Dataset, node: dbast.OutputOp):
    input_table = node.input
    output_table = node.output

    if output_table.columns and output_table.columns[0].metadata.metadata.get("is_star", False):
        new_columns = []
        for in_col in input_table.columns:
            col_attr = dbast.DBColumnAttr(dbast.Metadata())
            col_attr.metadata.metadata["name"] = in_col.metadata.metadata["name"]
            col_attr.metadata.metadata["dtype"] = in_col.metadata.metadata["dtype"]
            col_attr.metadata.metadata["nrows"] = in_col.metadata.metadata["nrows"]
            col_attr.metadata.metadata["is_star"] = False
            new_columns.append(col_attr)
        output_table.columns = new_columns
        output_table.metadata.metadata["ncols"] = len(input_table.columns)
        output_table.metadata.metadata["nrows"] = input_table.metadata.metadata.get("nrows", 0)
    else:
        for col_attr, in_col_attr in zip(output_table.columns, input_table.columns):
            col_attr.metadata.metadata["dtype"] = in_col_attr.metadata.metadata["dtype"]
            col_attr.metadata.metadata["nrows"] = in_col_attr.metadata.metadata["nrows"]


def resolve_dbcmpi_op(dataset: Dataset, node: dbast.CmpIOp):
    resolve_dbcolumn(dataset, node.lhs)
    node.output.dtype = "bool"


NODE_RESOLVE_MAP = {
    dbast.DBTable: resolve_dbtable,
    dbast.DBColumn: resolve_dbcolumn,
    dbast.ScanOp: resolve_dbscan_op,
    dbast.FilterOp: resolve_dbfilter_op,
    dbast.OutputOp: resolve_dboutput_op,
    dbast.CmpIOp: resolve_dbcmpi_op
}

