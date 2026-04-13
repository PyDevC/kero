#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/tensor.h>

#include <memory>

// Descriptor for a 2D MemRef matching ranked_tensor<100x1000xf32>
struct MemRef2D {
    float *allocated;
    float *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
};

class ArrowTableTensorFormat {
    private:
    public:
    /// Convert a column from arrow table to tensor of floating point
    std::shared_ptr<arrow::Tensor> arrowToTensor(
        std::shared_ptr<arrow::Table> table,
        const std::string& columnName,
        const std::vector<int64_t>& shape) {
        // TODO: Add the type option to params so that you can change the type
        // of the tensor from function call.
        auto column = table->GetColumnByName(columnName);
        // TODO: Check if you even need to do this or not.
        if (!column) { return nullptr; }

        std::shared_ptr<arrow::Array> fullArray;
        arrow::Result<std::shared_ptr<arrow::Array>> combinedResult =
            arrow::Concatenate(column->chunks());

        if (!combinedResult.ok()) { return nullptr; }

        fullArray = combinedResult.ValueOrDie();
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

    void executeQuery(MemRef2D input, MemRef2D *output, 
                     void (*mlir_func)(float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t)) {
        mlir_func(
            input.allocated, input.aligned, input.offset, 
            input.sizes[0], input.sizes[1], 
            input.strides[0], input.strides[1]
        );
    }
};
