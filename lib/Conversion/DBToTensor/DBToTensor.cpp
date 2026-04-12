#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBOps.h"
#include "Dialect/DB/IR/DBTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_DBTOTENSOR
#include "Conversion/DBToTensor/DBToTensor.h.inc"

namespace mlir {
namespace db {

class DBToTensorTypeConverter : public TypeConverter {
    public:
    DBToTensorTypeConverter(MLIRContext* context) {
        auto dynshape = mlir::ShapedType::kDynamic;

        /// Keep the type same for anonymous type
        addConversion([](Type type) { return type; });

        /// Convert ResultType to memref<?xf32>
        addConversion([dynshape, context](db::ResultType restype) {
            auto f32 = mlir::Float32Type::get(context);
            return mlir::MemRefType::get({dynshape}, f32);
        });

        /// Convert TableType to memref<?x?xf32>
        addConversion([dynshape, context](db::TableType tbltype) {
            auto f32 = mlir::Float32Type::get(context);
            return mlir::MemRefType::get({dynshape, dynshape}, f32);
        });

        /// Convert ColumnType to memref<1x?xf32>
        addConversion([dynshape, context](db::ColumnType coltype) {
            auto f32 = mlir::Float32Type::get(context);
            return mlir::MemRefType::get({1, dynshape}, f32);
        });

        /// Convert RowType to IndexType
        addConversion([context](db::RowType rowtype) {
            return mlir::IndexType::get(context);
        });
    }
};

class ConvertDBScan : public OpConversionPattern<ScanOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ScanOp Op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        auto dynshape = mlir::ShapedType::kDynamic;
        auto dim = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 0);
        auto newTensorType = RankedTensorType::get({dynshape, dynshape}, rewriter.getF32Type());
        auto newScanTensor = tensor::EmptyOp::create(rewriter, Op.getLoc(), newTensorType.getShape(), newTensorType.getElementType(), SmallVector<mlir::Value, 2>({dim, dim}));
        rewriter.replaceOp(Op, newScanTensor);
        return success();
    }
};

struct DBToTensor : ::impl::DBToTensorBase<DBToTensor> {
    using DBToTensorBase::DBToTensorBase;
    void runOnOperation() override {
        MLIRContext* context = &getContext();
        DBToTensorTypeConverter typeConverter(context);
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<tensor::TensorDialect>();
        target.addIllegalDialect<db::DBDialect>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp Op) {
            return typeConverter.isSignatureLegal(Op.getFunctionType());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp Op) {
            return typeConverter.isLegal(Op.getOperandTypes());
        });

        RewritePatternSet patterns(context);
        patterns.add<ConvertDBScan>(typeConverter, context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<mlir::Pass> createDBToTensor() {
    return std::make_unique<DBToTensor>();
}

} // namespace db
} // namespace mlir
