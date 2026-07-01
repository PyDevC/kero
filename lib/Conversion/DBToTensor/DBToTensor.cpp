#include "Conversion/DBToTensor/DBToTensor.h"
#include "Dialect/DB/IR/DBDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"

#include "llvm/ADT/STLExtras.h"
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
            return type.getDtype();
        });

        addConversion([](TableType type, llvm::SmallVectorImpl<Type>& results) {
            auto columns = type.getColumns();
            for (auto col : columns) {
                auto shape = col.getNrows();
                auto dtype = col.getDtype();

                if (shape < 0) {
                    results.push_back(RankedTensorType::get({ShapedType::kDynamic}, dtype));
                } else {
                    results.push_back(RankedTensorType::get({shape}, dtype));
                }
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
        // Create Mask Op
        auto identityType = rewriter.getI1Type();
        auto nrows = Op.getTable().getType().getNrows();
        auto output = tensor::EmptyOp::create(rewriter, Op.getLoc(), {nrows}, identityType);
        auto blockArguments = adaptor.getOperands().front();

        auto loopMap = rewriter.getDimIdentityMap();
        SmallVector<AffineMap> indexingMaps(blockArguments.size() + 1, loopMap);
        SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};

        auto maskOp = linalg::GenericOp::create(
            rewriter, Op.getLoc(),
            TypeRange{output.getType()},
            blockArguments,
            ValueRange{output},
            indexingMaps,
            iteratorTypes,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                Block* oldBlock = &Op.getRegion().front();
                IRMapping mapping;
                for (auto [oldArg, newArg] : llvm::zip(oldBlock->getArguments(), blockArgs.drop_back())) {
                    mapping.map(oldArg, newArg);
                }

                for (auto& innerOp : oldBlock->getOperations()) {
                    nestedBuilder.clone(innerOp, mapping);
                }
            });

        // Filter Yield Op will take care of the yield

        // Apply the mask tensor to the new tensors
        // Try applying SCF.ForOp to apply the mask on all tensors
        auto lowerBound = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 0);
        auto upperBound = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), nrows);
        auto stepLoop = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 1);
        auto zeroOp = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 0);
        auto newTensorSizeLoop = scf::ForOp::create(
            rewriter, Op.getLoc(),
            lowerBound,
            upperBound,
            stepLoop,
            ValueRange{zeroOp},
            [&](OpBuilder& builder, Location loc, Value outerIter, ValueRange iterArgs) {
                auto maskVal = tensor::ExtractOp::create(builder, loc, maskOp.getResult(0), outerIter);
                auto nextCount = scf::IfOp::create(builder, loc,
                                                   TypeRange{stepLoop.getType()},
                                                   maskVal, true, true);
                // If Block
                {
                    OpBuilder::InsertionGuard insertionGuard(builder);
                    builder.setInsertionPointToStart(nextCount.thenBlock());
                    auto updatedIndex = arith::AddIOp::create(builder, loc,
                                                              stepLoop.getType(),
                                                              stepLoop, iterArgs.front());
                    scf::YieldOp::create(builder, loc, updatedIndex.getResult());
                }

                // Else Block
                {
                    OpBuilder::InsertionGuard insertionGuard(builder);
                    builder.setInsertionPointToStart(nextCount.elseBlock());
                    scf::YieldOp::create(builder, loc, iterArgs.front());
                }

                scf::YieldOp::create(builder, loc, nextCount.getResults());
            });

        // Create new empty tensor from newTensorSizeLoop result
        auto inputTensors = adaptor.getOperands().front();
        auto numInputTensors = inputTensors.size();
        SmallVector<Value> newEmptyTensors{};
        SmallVector<Type> newEmptyTensorsType{};

        auto columns = Op.getTable().getType().getColumns();

        for (size_t i{}; i < numInputTensors; ++i) {
            auto columnType = columns[i].getDtype();
            auto emptyTensor = tensor::EmptyOp::create(rewriter, Op.getLoc(),
                                                       {newTensorSizeLoop.getResult(0)},
                                                       columnType);
            newEmptyTensors.push_back(emptyTensor);
            newEmptyTensorsType.push_back(columnType);
        }

        newEmptyTensors.push_back(zeroOp);

        // Apply the SCF.ForOp to write values to newEmptyTensors
        ValueRange iterArgs{newEmptyTensors};
        auto applyMaskToTensorsLoop = scf::ForOp::create(
            rewriter, Op.getLoc(),
            lowerBound, upperBound, stepLoop,
            iterArgs,
            [&](OpBuilder& builder, Location loc, Value iter, ValueRange innerIterArgs) {
                auto maskVal = tensor::ExtractOp::create(builder, loc, maskOp.getResult(0), iter);
                auto ifResultTypes = innerIterArgs.getTypes();
                auto finalTensor = scf::IfOp::create(builder, loc,
                                                     ifResultTypes,
                                                     maskVal, true, true);

                // If block
                {
                    OpBuilder::InsertionGuard insertionGuard(builder);
                    builder.setInsertionPointToStart(finalTensor.thenBlock());
                    SmallVector<Value> innerIfOpResult{};
                    for (size_t i{}; i < numInputTensors; ++i) {
                        auto tensorVal = tensor::ExtractOp::create(builder, loc,
                                                                   inputTensors[i], iter);
                        auto insertValue = tensor::InsertOp::create(builder, loc,
                                                                    tensorVal,
                                                                    innerIterArgs[i],
                                                                    innerIterArgs.back());
                        innerIfOpResult.push_back(insertValue);
                    }
                    auto updatedIndex = arith::AddIOp::create(builder, loc,
                                                              stepLoop.getType(),
                                                              stepLoop,
                                                              innerIterArgs.back());
                    innerIfOpResult.push_back(updatedIndex);
                    scf::YieldOp::create(builder, loc, innerIfOpResult);
                }

                // Else Block
                {
                    OpBuilder::InsertionGuard insertionGuard(builder);
                    builder.setInsertionPointToStart(finalTensor.elseBlock());
                    scf::YieldOp::create(builder, loc, innerIterArgs);
                }
                scf::YieldOp::create(builder, loc, finalTensor.getResults());
            });

        SmallVector<Value> filteredTensors{};
        for (size_t i{}; i < numInputTensors; ++i) {
            filteredTensors.push_back(applyMaskToTensorsLoop.getResult(i));
        }

        auto castOp = mlir::UnrealizedConversionCastOp::create(
            rewriter,
            Op.getLoc(),
            TypeRange{Op.getType()},
            filteredTensors);

        rewriter.replaceOp(Op, castOp.getResults());
        return success();
    }
};

class FilterYieldOpLowering : public OpConversionPattern<FilterYieldOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterYieldOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        if (adaptor.getOperands().empty()) { return failure(); }

        auto yieldValue = adaptor.getOperands()[0].front();

        if (auto castOp = yieldValue.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
            yieldValue = castOp.getInputs().front();
        }

        rewriter.replaceOpWithNewOp<linalg::YieldOp>(Op, yieldValue);
        return success();
    }
};

class CmpIOpLowering : public OpConversionPattern<CmpIOp> {
    private:
    arith::CmpIPredicate translatePredicate(db::CmpIPredicate pred) const {
        switch (pred) {
            case CmpIPredicate::lt: return arith::CmpIPredicate::slt;
            case CmpIPredicate::eq: return arith::CmpIPredicate::eq;
            case CmpIPredicate::gt: return arith::CmpIPredicate::sgt;
            case CmpIPredicate::lte: return arith::CmpIPredicate::sle;
            case CmpIPredicate::gte: return arith::CmpIPredicate::sge;
            case CmpIPredicate::neq: return arith::CmpIPredicate::ne;
        }
    }

    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CmpIOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];

        auto newPred = translatePredicate(Op.getPredicate());
        auto compare = arith::CmpIOp::create(rewriter, Op.getLoc(),
                                             rewriter.getI1Type(),
                                             newPred,
                                             lhs[0],
                                             rhs[0]);

        rewriter.replaceOp(Op, compare.getResult());
        return success();
    }
};

class OutputOpLowering : public OpConversionPattern<OutputOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(OutputOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const final {
        auto tableType = Op.getTable().getType();
        auto selectAttr = Op.getSelectAttr();
        auto tableColumns = tableType.getColumns();
        auto inputTensors = adaptor.getOperands();

        llvm::DenseSet<StringRef> selectedNames;
        for (auto select : selectAttr) {
            auto colName = cast<StringAttr>(select).getValue();
            selectedNames.insert(colName);
        }

        SmallVector<Value> outputTensors{};
        for (auto [idx, valRange] : llvm::enumerate(inputTensors)) {
            if (selectedNames.contains(tableColumns[idx].getName())) {
                outputTensors.push_back(valRange.front());
            }
        }

        rewriter.replaceOpWithMultiple(Op, outputTensors);
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
                               scf::SCFDialect,
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
        patterns.add<CmpIOpLowering>(converter, ctx);
        patterns.add<FilterOpLowering, FilterYieldOpLowering>(converter, ctx);
        patterns.add<OutputOpLowering>(converter, ctx);

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
