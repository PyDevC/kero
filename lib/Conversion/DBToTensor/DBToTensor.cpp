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

namespace mlir {
namespace db {

#define GEN_PASS_DEF_DBTOTENSOR
#include "Conversion/DBToTensor/DBToTensor.h.inc"

/// DBToTensor TypeConverter which converts the single db-dialect type
/// to MemRefType making it easier to be mapped with Apache Arrow Column
/// Layout
class DBToTensorTypeConverter : public TypeConverter {
    public:
    explicit DBToTensorTypeConverter(MLIRContext* context) {
        addConversion([](Type type) { return type; });

        // !db.result → tensor<?x?xf32>
        addConversion([&](db::ResultType) -> Type {
            return RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                         Float32Type::get(context));
        });

        // !db.table<"t"> → tensor<?x?xf32>
        addConversion([&](db::TableType) -> Type {
            return RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                         Float32Type::get(context));
        });

        // !db.column<"t","c",T> → tensor<1x?xT>
        addConversion([&](db::ColumnType colType) -> Type {
            return RankedTensorType::get({1, ShapedType::kDynamic},
                                         colType.getDtype());
        });

        // !db.row → index
        addConversion([&](db::RowType) -> Type {
            return IndexType::get(context);
        });
    }
};

class ConvertDBScan : public OpConversionPattern<ScanOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(op, adaptor.getTable());
        return success();
    }
};

class ConvertDBReturn : public OpConversionPattern<ReturnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};


class ConvertDBFilter : public OpConversionPattern<FilterOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        Value inputTensor = adaptor.getInput();
        auto tensorType = llvm::dyn_cast<RankedTensorType>(inputTensor.getType());
        if (!tensorType)
            return rewriter.notifyMatchFailure(op, "Expected ranked tensor input");

        Location loc = op.getLoc();

        Value mask = generateMaskTensor(inputTensor, op.getRegion(), rewriter, loc);

        Value emptyOut = buildDynamicEmpty(rewriter, loc, tensorType, inputTensor);

        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(tensorType.getElementType()));
        Value zeroFilled =
            linalg::FillOp::create(rewriter, loc,
                                   ValueRange{zero}, ValueRange{emptyOut})
                .getResult(0);

        auto map = rewriter.getMultiDimIdentityMap(tensorType.getRank());
        SmallVector<AffineMap> maps = {map, map, map};
        SmallVector<utils::IteratorType> iterators(
            tensorType.getRank(), utils::IteratorType::parallel);

        auto selectOp = rewriter.create<linalg::GenericOp>(
            loc, tensorType,
            /*inputs=*/ValueRange{inputTensor, mask},
            /*outputs=*/ValueRange{zeroFilled},
            maps, iterators,
            [&](OpBuilder& b, Location innerLoc, ValueRange args) {
                // args[0] = input element, args[1] = mask bit, args[2] = zero
                Value selected =
                    arith::SelectOp::create(b, innerLoc, args[1], args[0], args[2]);
                linalg::YieldOp::create(b, innerLoc, selected);
            });

        rewriter.replaceOp(op, selectOp.getResult(0));
        return success();
    }

    private:
    Value generateMaskTensor(Value inputTensor, Region& filterRegion,
                             ConversionPatternRewriter& rewriter,
                             Location loc) const {
        auto tensorType = cast<RankedTensorType>(inputTensor.getType());

        auto i1Type = rewriter.getI1Type();
        auto maskType = RankedTensorType::get(tensorType.getShape(), i1Type);

        Value initMask = buildDynamicEmpty(rewriter, loc, maskType, inputTensor);

        auto map = rewriter.getMultiDimIdentityMap(tensorType.getRank());
        SmallVector<AffineMap> maps = {map, map};
        SmallVector<utils::IteratorType> iterators(
            tensorType.getRank(), utils::IteratorType::parallel);

        auto maskOp = rewriter.create<linalg::GenericOp>(
            loc, maskType,
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{initMask},
            maps, iterators,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
                Value rowIV =
                    nestedBuilder.create<linalg::IndexOp>(nestedLoc, 1);

                IRMapping mapper;
                mapper.map(filterRegion.getArgument(0), rowIV);

                for (auto& nestedOp : filterRegion.front().without_terminator())
                    nestedBuilder.clone(nestedOp, mapper);

                auto returnOp =
                    cast<db::ReturnOp>(filterRegion.front().getTerminator());
                Value result = mapper.lookupOrDefault(returnOp.getInput());
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
            });

        return maskOp.getResult(0);
    }

    static Value buildDynamicEmpty(OpBuilder& builder, Location loc,
                                   RankedTensorType targetType,
                                   Value referenceSource) {
        ArrayRef<int64_t> shape = targetType.getShape();
        SmallVector<Value> dynSizes;

        for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
            if (shape[i] == ShapedType::kDynamic) {
                Value dim = tensor::DimOp::create(builder, loc, referenceSource,
                                                          builder.create<arith::ConstantIndexOp>(loc, i));
                dynSizes.push_back(dim);
            }
        }

        return tensor::EmptyOp::create(builder, loc, shape,
                                       targetType.getElementType(), dynSizes);
    }
};

class ConvertDBGetCol : public OpConversionPattern<GetColumnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GetColumnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        Location loc = op.getLoc();
        Value rowIndex = adaptor.getRow();
        Value columnTensor = adaptor.getColumn();

        if (!isa<IndexType>(rowIndex.getType())) {
            rowIndex = arith::IndexCastOp::create(rewriter, loc,
                                                  rewriter.getIndexType(), rowIndex);
        }

        Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
        SmallVector<Value> indices{zeroIdx, rowIndex};
        Value extracted =
            tensor::ExtractOp::create(rewriter, loc, columnTensor, indices);

        rewriter.replaceOp(op, extracted);
        return success();
    }
};

class ConvertDBProject : public OpConversionPattern<ProjectOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ProjectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(op, adaptor.getFilter());
        return success();
    }
};

struct DBToTensor : impl::DBToTensorBase<DBToTensor> {
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

        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return typeConverter.isLegal(op.getOperandTypes());
        });
        target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
            return typeConverter.isLegal(op);
        });

        RewritePatternSet patterns(context);
        patterns.add<ConvertDBScan,
                     ConvertDBReturn,
                     ConvertDBFilter,
                     ConvertDBGetCol,
                     ConvertDBProject>(typeConverter, context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns, typeConverter);
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        populateCallOpTypeConversionPattern(patterns, typeConverter);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace db
} // namespace mlir
