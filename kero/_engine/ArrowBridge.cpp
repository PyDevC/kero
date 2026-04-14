#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/tensor.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

// Descriptor for a 2D MemRef matching ranked_tensor<100x1000xf32>
struct MemRef2D {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

class ArrowTableTensorFormat {
public:
    ArrowTableTensorFormat() = default;

    /// Convert a column from arrow table to tensor of floating point
    std::shared_ptr<arrow::Tensor> arrowToTensor(
        std::shared_ptr<arrow::Table> table,
        const std::string& columnName,
        const std::vector<int64_t>& shape) {
        
        auto column = table->GetColumnByName(columnName);
        if (!column) { return nullptr; }

        arrow::Result<std::shared_ptr<arrow::Array>> combinedResult =
            arrow::Concatenate(column->chunks());

        if (!combinedResult.ok()) { return nullptr; }

        auto fullArray = combinedResult.ValueOrDie();
        auto floatArray = std::static_pointer_cast<arrow::FloatArray>(fullArray);
        std::shared_ptr<arrow::Buffer> columnBuffer = floatArray->values();

        std::shared_ptr<arrow::DataType> float32 = arrow::float32();
        return arrow::Tensor::Make(float32, columnBuffer, shape).ValueOrDie();
    }

    MemRef2D tensorToMemRef(std::shared_ptr<arrow::Tensor> tensor) {
        auto data_ptr = reinterpret_cast<const float*>(tensor->data()->data());
        
        MemRef2D descriptor;
        descriptor.allocated = const_cast<float*>(data_ptr);
        descriptor.aligned = const_cast<float*>(data_ptr);
        descriptor.offset = 0;
        
        descriptor.sizes[0] = tensor->shape()[0];
        descriptor.sizes[1] = tensor->shape()[1];
        
        descriptor.strides[0] = tensor->shape()[1];
        descriptor.strides[1] = 1;
        
        return descriptor;
    }
};

NB_MODULE(_arrow_bridge, m) {
    // We bind the MemRef2D struct so Python can handle the return type
    nb::class_<MemRef2D>(m, "MemRef2D")
        .def_rw("offset", &MemRef2D::offset)
        .def_prop_ro("sizes", [](const MemRef2D &m) { 
            return std::vector<int64_t>{m.sizes[0], m.sizes[1]}; 
        })
        .def_prop_ro("strides", [](const MemRef2D &m) { 
            return std::vector<int64_t>{m.strides[0], m.strides[1]}; 
        });

    nb::class_<ArrowTableTensorFormat>(m, "ArrowTableTensorFormat")
        .def(nb::init<>())
        .def("arrow_to_tensor", &ArrowTableTensorFormat::arrowToTensor, 
             "table"_a, "column_name"_a, "shape"_a)
        .def("tensor_to_memref", &ArrowTableTensorFormat::tensorToMemRef, "tensor"_a);
}
