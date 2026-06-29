#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace db {

#define GEN_PASS_DEF_DBTOTENSOR
#include "Conversion/DBToTensor/DBToTensor.h.inc"

class DBTypeToTensorConverter : public TypeConverter {
    public:
    DBTypeToTensorConverter(MLIRContext* ctx) {
        addConversion([](Type type) { return type; });

        addConversion([](ColumnType type) {
            auto dtype = type.getType();
            auto shape = ShapedType::kDynamic;
            return RankedTensorType::get(shape, dtype);
        });

        addConversion([](TableType type, llvm::SmallVectorImpl<Type>& results) {
            auto columns = type.getColumns();
            for (auto col : columns) {
                auto shape = col.getNrows();
                auto dtype = col.getType();

                results.push_back(RankedTensorType::get({shape}, dtype));
            }
            return success();
        });
    }
};

class ScanOpLowering : public OpConversionPattern<ScanOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ScanOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        auto newOp = adaptor.getOperands();
        rewriter.replaceOpWithMultiple(Op, newOp);
        return success();
    }
};

class FilterOpLowering : public OpConversionPattern<FilterOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        auto nrows = Op.getTable().getType().getNrows();
        auto i1 = IntegerType::get(getContext(), 1);
        auto output = tensor::EmptyOp::create(rewriter, Op.getLoc(), {nrows}, i1);

        auto inputs = adaptor.getRegions().front()->getArguments();

        auto identityMap = AffineMap::get(1, 0, rewriter.getAffineDimExpr(0), getContext());
        SmallVector<AffineMap> idxMap{};
        for (size_t i{}; i < inputs.size(); ++i) {
            idxMap.push_back(identityMap);
        }
        idxMap.push_back(identityMap);

        auto initRank = cast<RankedTensorType>(inputs.front().getType()).getRank();
        SmallVector<utils::IteratorType> iteratorType(initRank, utils::IteratorType::parallel);

        auto filterLoop = linalg::GenericOp::create(rewriter, Op.getLoc(),
                                                    output.getType(),
                                                    inputs,
                                                    ValueRange{output},
                                                    idxMap,
                                                    iteratorType,
                                                    [&](OpBuilder& builder, Location loc, ValueRange blockArgs) {
                                                    });

        rewriter.replaceOp(Op, filterLoop);
        return success();
    }
};

class CmpIOPLowering : public OpConversionPattern<CmpIOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CmpIOp Op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        return success();
    }
};

class OutputOpLowering : public OpConversionPattern<OutputOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(OutputOp Op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        return success();
    }
};

class DecomposeFuncOp : public OpConversionPattern<func::FuncOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(func::FuncOp Op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        auto converter = *getTypeConverter();
        TypeConverter::SignatureConversion sigConversion(Op.getFunctionType().getNumInputs());

        SmallVector<Type> newArgs{};
        auto funcInputs = Op.getFunctionType().getInputs();
        for (auto [idx, arg] : llvm::enumerate(funcInputs)) {
            SmallVector<Type> expanedArgs{};

            if (failed(converter.convertType(arg, expanedArgs))) {
                return failure();
            }

            sigConversion.addInputs(idx, expanedArgs);
            newArgs.append(expanedArgs.begin(), expanedArgs.end());
        }

        SmallVector<Type> newResults{};
        for (auto resType : Op.getFunctionType().getResults()) {
            auto result = converter.convertType(resType);
            if (!result) { return failure(); }
            newResults.push_back(result);
        }

        auto newFuncType = FunctionType::get(Op.getContext(), newArgs, newResults);
        auto newFunc = func::FuncOp::create(rewriter, Op.getLoc(), Op.getName(), newFuncType);

        newFunc.setVisibility(Op.getVisibility());
        rewriter.inlineRegionBefore(Op.getRegion(), newFunc.getRegion(), newFunc.end());
        rewriter.applySignatureConversion(&newFunc.getBody().front(), sigConversion);

        rewriter.eraseOp(Op);
        return success();
    }
};

struct DBToTensor : impl::DBToTensorBase<DBToTensor> {
    using DBToTensorBase::DBToTensorBase;

    void runOnOperation() override {
        auto module = getOperation();
        auto ctx = &getContext();
        ConversionTarget target(*ctx);
        DBTypeToTensorConverter converter(ctx);

        target.addLegalDialect<mlir::BuiltinDialect,
                               func::FuncDialect,
                               tensor::TensorDialect,
                               linalg::LinalgDialect,
                               arith::ArithDialect>();

        target.addIllegalDialect<db::DBDialect>();

        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return converter.isSignatureLegal(op.getFunctionType());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return converter.isLegal(op.getOperandTypes());
        });
        target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
            return converter.isLegal(op);
        });

        RewritePatternSet patterns(ctx);
        patterns.add<DecomposeFuncOp>(converter, ctx);
        patterns.add<ScanOpLowering>(converter, ctx);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        populateReturnOpTypeConversionPattern(patterns, converter);
        populateCallOpTypeConversionPattern(patterns, converter);

        if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace db
} // namespace mlir
