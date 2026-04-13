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

/// DBToTensor TypeConverter which converts the single db-dialect type
/// to MemRefType making it easier to be mapped with Apache Arrow Column
/// Layout
class DBToTensorTypeConverter : public TypeConverter {
    public:
    DBToTensorTypeConverter(MLIRContext* context) {
        auto f32 = mlir::Float32Type::get(context);

        /// Keep the type same for anonymous type
        addConversion([](Type type) { return type; });

        /// Convert ResultType to memref<100x1000xf32>
        /// The Result should always be compatible with Table.
        addConversion([f32](db::ResultType restype) {
            return mlir::RankedTensorType::get({100, 1000}, f32);
        });

        /// Convert TableType to memref<100x1000xf32>
        addConversion([f32](db::TableType tbltype) {
            return mlir::RankedTensorType::get({100, 1000}, f32);
        });

        /// Convert ColumnType to memref<1x1000xf32>
        addConversion([](db::ColumnType coltype) {
            auto dtype = coltype.getDtype();
            return mlir::RankedTensorType::get({1, 1000}, dtype);
        });

        /// Convert RowType to IndexType
        addConversion([context](db::RowType rowtype) {
            return mlir::IndexType::get(context);
        });

        /// Convert Tensor to DBColumn
        addSourceMaterialization([](OpBuilder& builder, db::ColumnType type,
                                    ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return nullptr;
            return db::GetColumnOp::create(builder, loc, type, inputs[0]);
        });
    }
};

/// Convert the ScanOp to Table Tensor
class ConvertDBScan : public OpConversionPattern<ScanOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ScanOp Op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(Op, adaptor.getTable());
        return success();
    }
};

/// Convert DB.ReturnOp to func.ReturnOp Since both are the same thing
class ConvertDBReturn : public OpConversionPattern<ReturnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ReturnOp Op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(Op, adaptor.getOperands());
        return success();
    }
};

/// DBFilter Applies tensor.SelectOp to extract specific index elements
/// It unpacks the Region and applies the predicate logic
class ConvertDBFilter : public OpConversionPattern<FilterOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        Value inputTensor = adaptor.getInput();
        auto tensorType = llvm::dyn_cast<RankedTensorType>(inputTensor.getType());

        if (!tensorType) {
            return rewriter.notifyMatchFailure(op, "Expected ranked tensor type");
        }

        Location loc = op.getLoc();

        Value mask = generateMaskedTensor(inputTensor, op.getRegion(), rewriter, loc);

        Value zeroTensor = tensor::EmptyOp::create(
            rewriter, loc, tensorType.getShape(), tensorType.getElementType());
        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(tensorType.getElementType()));
        Value zeroFilled = linalg::FillOp::create(
                               rewriter, loc, ValueRange{zero}, ValueRange{zeroTensor})
                               .getResult(0);

        auto map = rewriter.getMultiDimIdentityMap(tensorType.getRank());
        SmallVector<AffineMap> maps = {map, map, map};
        SmallVector<utils::IteratorType> iterators(
            tensorType.getRank(), utils::IteratorType::parallel);

        Value filteredTensor = linalg::GenericOp::create(
                                   rewriter, loc, tensorType,
                                   /*inputs=*/ValueRange{inputTensor, mask},
                                   /*outputs=*/zeroFilled,
                                   maps, iterators,
                                   [&](OpBuilder& b, Location loc, ValueRange args) {
                                       Value inputValue = args[0];
                                       Value maskValue = args[1];
                                       Value zeroValue = args[2];

                                       Value result = arith::SelectOp::create(b, loc, maskValue, inputValue, zeroValue);
                                       linalg::YieldOp::create(b, loc, result);
                                   })
                                   .getResult(0);

        rewriter.replaceOp(op, filteredTensor);
        return success();
    }

    private:
    Value generateMaskedTensor(Value inputTensor, Region& filterRegion,
                               ConversionPatternRewriter& rewriter, Location loc) const {
        auto tensorType = cast<RankedTensorType>(inputTensor.getType());
        auto shape = tensorType.getShape();
        auto i1Type = rewriter.getI1Type();
        auto maskedType = RankedTensorType::get(shape, i1Type);

        Value initMask = tensor::EmptyOp::create(rewriter, loc, shape, i1Type);

        auto map = rewriter.getMultiDimIdentityMap(shape.size());
        SmallVector<AffineMap> maps = {map, map};
        SmallVector<utils::IteratorType> iterators(shape.size(), utils::IteratorType::parallel);

        auto genericOp = linalg::GenericOp::create(
            rewriter,
            loc,
            maskedType,
            /*inputs=*/inputTensor,
            /*outputs=*/initMask,
            maps,
            iterators,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
                Value rowIV = linalg::IndexOp::create(nestedBuilder, nestedLoc, 1); // dim 1 = column index = row id

                IRMapping mapper;
                mapper.map(filterRegion.getArgument(0), rowIV);

                for (auto& nestedOp : filterRegion.front().without_terminator()) {
                    nestedBuilder.clone(nestedOp, mapper);
                }

                auto returnOp = cast<db::ReturnOp>(filterRegion.front().getTerminator());
                Value result = mapper.lookupOrDefault(returnOp.getInput());
                linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
            });

        return genericOp.getResult(0);
    }
};

/// Convert DBGetColOp to tensor.extract
class ConvertDBGetCol : public OpConversionPattern<GetColumnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(GetColumnOp Op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        Value rowIndex = adaptor.getRow();
        Value columnTensor = adaptor.getColumn();

        Location loc = Op.getLoc();

        // rowIndex comes from RowType -> IndexType conversion, must be index
        if (!isa<IndexType>(rowIndex.getType())) {
            rowIndex = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), rowIndex);
        }

        Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
        SmallVector<Value> indices{zeroIndex, rowIndex};

        Value extractedValue = tensor::ExtractOp::create(rewriter, loc, columnTensor, indices);

        rewriter.replaceOp(Op, extractedValue);
        return success();
    }
};

/// Convert DBProjectOp to apply DBFilterOp since this is the 
/// way to call the filter over multiple places
class ConvertDBProject : public OpConversionPattern<ProjectOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(ProjectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(op, adaptor.getFilter());
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
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addIllegalDialect<db::DBDialect>();
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp Op) {
            return typeConverter.isSignatureLegal(Op.getFunctionType());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp Op) {
            return typeConverter.isLegal(Op.getOperandTypes());
        });
        target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
            return typeConverter.isLegal(op);
        });

        RewritePatternSet patterns(context);
        patterns.add<ConvertDBScan, ConvertDBReturn, ConvertDBFilter, ConvertDBGetCol, ConvertDBProject>(typeConverter, context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
        populateCallOpTypeConversionPattern(patterns, typeConverter);

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
