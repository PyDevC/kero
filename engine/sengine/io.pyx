from collections import OrderedDict

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared, make_unique
from cython.operator cimport dereference as deref, postincrement
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen

from enum import IntEnum

import numpy as np
import pandas as pd

import cudf

from cudf._lib cimport *
from cudf._lib.types import np_to_cudf_types, cudf_to_np_types
from cudf._lib.cpp.types cimport type_id
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.cpp.io.types cimport compression_type

from sengine.io cimport cio
from sengine.io.cio cimport *
from cpython.ref cimport PyObject
from cython.operator cimport dereference, postincrement

from cudf._lib.table cimport Table as CudfXxTable

ctypedef int32_t underlying_type_t_compression

class Compression(IntEnum):
    INFER = (
        <underlying_type_t_compression> compression_type.AUTO
    )
    SNAPPY = (
        <underlying_type_t_compression> compression_type.SNAPPY
    )
    GZIP = (
        <underlying_type_t_compression> compression_type.GZIP
    )
    BZ2 = (
        <underlying_type_t_compression> compression_type.BZIP2
    )
    BROTLI = (
        <underlying_type_t_compression> compression_type.BROTLI
    )
    ZIP = (
        <underlying_type_t_compression> compression_type.ZIP
    )
    XZ = (
        <underlying_type_t_compression> compression_type.XZ
    )

class KeroError(Exception):
    """Base class errors."""
cdef public PyObject * KeroError_ = <PyObject *> KeroError

class InitializeError(BlazingError):
    """Initialization Error."""
cdef public PyObject * InitializeError_ = <PyObject *>InitializeError

class FinalizeError(BlazingError):
    """Finalize Error."""
cdef public PyObject * FinalizeError_ = <PyObject *>FinalizeError

class GetFreeMemoryError(BlazingError):
    """GetFreeMemory Error."""
cdef public PyObject * GetFreeMemoryError_ = <PyObject *>GetFreeMemoryError

class ResetMaxMemoryUsedError(BlazingError):
    """ResetMaxUsedMemoryError Error."""
cdef public PyObject * ResetMaxMemoryUsedError_ = <PyObject *>ResetMaxMemoryUsedError

class GetMaxMemoryUsedError(BlazingError):
    """GetMaxMemoryUsedError Error."""
cdef public PyObject * GetMaxMemoryUsedError_ = <PyObject *>GetMaxMemoryUsedError

class GetProductDetailsError(BlazingError):
    """GetProductDetails Error."""
cdef public PyObject * GetProductDetailsError_ = <PyObject *>GetProductDetailsError

class PerformPartitionError(BlazingError):
    """PerformPartitionError Error."""
cdef public PyObject * PerformPartitionError_ = <PyObject *>PerformPartitionError

class RunGenerateGraphError(BlazingError):
    """RunGenerateGraph Error."""
cdef public PyObject * RunGenerateGraphError_ = <PyObject *>RunGenerateGraphError

class RunExecuteGraphError(BlazingError):
    """RunExecuteGraph Error."""
cdef public PyObject * RunExecuteGraphError_ = <PyObject *>RunExecuteGraphError

class RunSkipDataError(BlazingError):
    """RunSkipData Error."""
cdef public PyObject *RunSkipDataError_ = <PyObject *>RunSkipDataError

class ParseSchemaError(BlazingError):
    """ParseSchema Error."""
cdef public PyObject * ParseSchemaError_ = <PyObject *>ParseSchemaError

class RegisterFileSystemHDFSError(BlazingError):
    """RegisterFileSystemHDFS Error."""
cdef public PyObject * RegisterFileSystemHDFSError_ = <PyObject *>RegisterFileSystemHDFSError

class RegisterFileSystemGCSError(BlazingError):
    """RegisterFileSystemGCS Error."""
cdef public PyObject * RegisterFileSystemGCSError_ = <PyObject *>RegisterFileSystemGCSError

class RegisterFileSystemS3Error(BlazingError):
    """RegisterFileSystemS3 Error."""
cdef public PyObject * RegisterFileSystemS3Error_ = <PyObject *>RegisterFileSystemS3Error

class RegisterFileSystemLocalError(BlazingError):
    """RegisterFileSystemLocal Error."""
cdef public PyObject * RegisterFileSystemLocalError_ = <PyObject *>RegisterFileSystemLocalError

class InferFolderPartitionMetadataError(BlazingError):
    """InferFolderPartitionMetadata Error."""
cdef public PyObject * InferFolderPartitionMetadataError_ = <PyObject *>InferFolderPartitionMetadataError


cdef cio.TableSchema parseSchemaPython(vector[string] files, 
                                       string file_format_hint, 
                                       vector[string] arg_keys, 
                                       vector[string] arg_values,
                                       vector[pair[string,type_id]] extra_columns, 
                                       bool ignore_missing_paths
) nogil except *:
    with nogil:
        return cio.parseSchema(files, file_format_hint, arg_keys, arg_values, extra_columns, ignore_missing_paths)

cdef void finalizePython(vector[int] ctx_tokens) nogil except +:
    with nogil:
        cio.finalize(ctx_tokens)

cdef size_t getFreeMemoryPython() nogil except *:
    with nogil:
        return cio.getFreeMemory()

cdef size_t getMaxMemoryUsedPython() nogil except *:
    with nogil:
        return cio.getMaxMemoryUsed()

cdef void resetMaxMemoryUsedPython() nogil except *:
    with nogil:
        cio.resetMaxMemoryUsed(0)

cdef map[string, string] getProductDetailsPython() nogil except *:
    with nogil:
        return cio.getProductDetails()

cdef vector[cio.FolderPartitionMetadata] inferFolderPartitionMetadataPython(string folder_path) nogil except *:
    with nogil:
        return cio.inferFolderPartitionMetadata(folder_path)

cdef class PyGeneralCache:

    def add_to_cache_with_meta(self,cudf_data,metadata):
        for key in metadata.keys():
          if key != "worker_ids":
            c_key = key.encode()
            metadata_map[c_key] = metadata[key].encode()
    # add cache to kernel keys to get faster execution 
    # theory only Don't try without test in other way. 
    # make sure that data doesn't get corupted

    def has_next_now(self,):
        return deref(self.c_cache).has_next_now()

    def add_to_cache(self,cudf_data):
        cdef vector[string] column_names
        for column_name in cudf_data:
            column_names.push_back(str.encode(column_name))

        with nogil:
            deref(self.c_cache).addToCache(move(table),message_id,1)

    def pull_from_cache(self):
        cdef unique_ptr[CacheData] cache_data
        with nogil:
            cache_data = blaz_move(deref(self.c_cache).pullCacheData())
        cdef MetadataDictionary metadata = deref(cache_data).getMetadata()
        cdef unique_ptr[BlazingTable] table = deref(cache_data).decache()

        metadata_temp = metadata.get_values()
        metadata_py = {}
        for key_val in metadata_temp:
            key = key_val.first.decode('utf-8')
            val = key_val.second.decode('utf-8')
            if(key == "worker_ids"):
                metadata_py[key] = val.split(",")
            else:
                metadata_py[key] = val


        decoded_names = []
        for i in range(deref(table).names().size()):
            decoded_names.append(deref(table).names()[i].decode('utf-8'))
        df = cudf.DataFrame(CudfXxTable.from_unique_ptr(blaz_move(deref(table).releaseCudfTable()), decoded_names)._data)
        df._rename_columns(decoded_names)
        return df, metadata_py

cpdef finalizeCaller(ctx_tokens: [int]):
    cdef vector[int] tks
    for ctx_token in ctx_tokens:
        tks.push_back(ctx_token)
    finalizePython(tks)

cpdef getFreeMemoryCaller():
    return getFreeMemoryPython()

cpdef getMaxMemoryUsedCaller():
    return getMaxMemoryUsedPython()

cpdef resetMaxMemoryUsedCaller():
    resetMaxMemoryUsedPython()

cpdef getProductDetailsCaller():
    my_map = getProductDetailsPython()
    cdef map[string,string].iterator it = my_map.begin()
    new_map = OrderedDict()
    while(it != my_map.end()):
        key = dereference(it).first
        key = key.decode('utf-8')

        value = dereference(it).second
        value = value.decode('utf-8')

        new_map[key] = value

        postincrement(it)

    return new_map

cpdef inferFolderPartitionMetadataCaller(folder_path):
    folderMetadataArr = inferFolderPartitionMetadataPython(folder_path.encode())

    return_array = []
    for metadata in folderMetadataArr:
        decoded_values = []
        for value in metadata.values:
            decoded_values.append(value.decode('utf-8'))

        return_array.append({
            'name': metadata.name.decode('utf-8'),
            'values': decoded_values,
            'data_type': <underlying_type_t_type_id>(metadata.data_type)
        })

    return return_array


cdef class PyGraph:
    cdef shared_ptr[cio.graph] ptr

    def get_kernel_output_cache(self,kernel_id, cache_id):
        cache = PyGeneralCache()
        cache.c_cache = deref(self.ptr).get_kernel_output_cache(int(kernel_id),str.encode(cache_id))
        return cache

    cpdef set_input_and_output_caches(self, PyBlazingCache input_cache, PyBlazingCache output_cache):
        deref(self.ptr).set_input_and_output_caches(input_cache.c_cache, output_cache.c_cache)

    cpdef query_is_complete(self):
        return deref(self.ptr).query_is_complete()

    cpdef get_progress(self):
        return deref(self.ptr).get_progress()

cpdef startExecuteGraphCaller(PyBlazingGraph graph, int ctx_token):

cpdef getTableScanInfoCaller(logicalPlan):
    temp = getTableScanInfoPython(str.encode(logicalPlan))

    table_names = [name.decode('utf-8') for name in temp.table_names]
    table_scans = [step.decode('utf-8') for step in temp.relational_algebra_steps]
    return table_names, table_scans

#Type conversions

cpdef np_to_cudf_types_int(dtype):
    return <underlying_type_t_type_id> ( np_to_cudf_types[dtype])

cpdef cudf_type_int_to_np_types(type_int):
    return cudf_to_np_types[<underlying_type_t_type_id> (type_int)]
