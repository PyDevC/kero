#include <nanobind/nanobind.h>

nanobind::dict query_dict_to_mlir(nanobind::dict &query_dict){
    nanobind::print(query_dict);
    return query_dict;
}

NB_MODULE(sql_to_mlir, m){
    m.def("query_dict_to_mlir", &query_dict_to_mlir);
}
