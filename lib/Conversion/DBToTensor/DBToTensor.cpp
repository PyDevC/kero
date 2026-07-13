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
                                  ConversionPatternRewriter& rewriter) const override {
        auto newOp = adaptor.getOperands();
        rewriter.replaceOpWithMultiple(Op, newOp);
        return success();
    }
};

linalg::GenericOp genericMaskedTensorOp(FilterOp Op, OpBuilder& builder,
                                        Location loc, ValueRange blockArgs) {
    auto identityType = builder.getI1Type();
    auto tensorSize = Op.getTable().getType().getNrows(); // Size of Mask Tensor
    auto maskedTensor = tensor::EmptyOp::create(
        builder, Op.getLoc(), {tensorSize}, identityType);

    auto loopMap = builder.getDimIdentityMap();
    // indexing Maps should be equal to blockArguments + 1 since
    // we have to input extra argument for maskedTensor
    SmallVector<AffineMap> indexingMaps(blockArgs.size() + 1, loopMap);
    SmallVector<utils::IteratorType> iteratorTypes{utils::IteratorType::parallel};

    auto maskOp = linalg::GenericOp::create(
        builder, loc,
        TypeRange{maskedTensor.getType()},
        blockArgs,
        ValueRange{maskedTensor},
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange nestedblockArgs) {
            Block* oldBlock = &Op.getRegion().front();
            IRMapping mapping;
            for (auto [oldArg, newArg] : llvm::zip(oldBlock->getArguments(), nestedblockArgs.drop_back())) {
                mapping.map(oldArg, newArg);
            }

            // required for recrusive lowering
            for (auto& innerOp : oldBlock->getOperations()) {
                nestedBuilder.clone(innerOp, mapping);
            }
        });

    return maskOp;
}

scf::ForOp scfNewTensorSizeLoopFromMask(
    FilterOp Op, OpBuilder& builder,
    Location loc, Value maskedTensor, arith::ConstantIndexOp lowerBound,
    arith::ConstantIndexOp upperBound, arith::ConstantIndexOp step) {
    auto newTensorSize = arith::ConstantIndexOp::create(builder, loc, 0);
    auto newTensorSizeLoop = scf::ForOp::create(
        builder, loc,
        lowerBound, upperBound,
        step,
        ValueRange{newTensorSize},
        [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iter, ValueRange nestedblockArgs) {
            // %mask_val = tensor.extract %maskeTensor[%iter] : tensor<upperBoundxtype>
            auto maskVal = tensor::ExtractOp::create(
                nestedBuilder, nestedLoc, maskedTensor, ValueRange{iter});
            auto ifOp = scf::IfOp::create(
                nestedBuilder, nestedLoc,
                TypeRange{newTensorSize.getType()}, maskVal, true, true);

            // if maskVal then
            //     newTensorSize_0 = newTensorSize + 1
            //     scf.yield newTensorSize_0
            //  else
            //     scf.yield
            //
            // Then Block
            {
                OpBuilder::InsertionGuard insertionGuard(nestedBuilder);
                nestedBuilder.setInsertionPointToStart(ifOp.thenBlock());

                auto newTensorSizeUpdate = arith::AddIOp::create(
                    nestedBuilder, nestedLoc,
                    newTensorSize.getType(), nestedblockArgs[0], step);

                scf::YieldOp::create(
                    nestedBuilder, nestedLoc, newTensorSizeUpdate.getResult());
            }

            // Else Block
            {
                OpBuilder::InsertionGuard insertionGuard(nestedBuilder);
                nestedBuilder.setInsertionPointToStart(ifOp.elseBlock());
                scf::YieldOp::create(nestedBuilder, nestedLoc, nestedblockArgs[0]);
            }
            scf::YieldOp::create(nestedBuilder, nestedLoc, ifOp.getResult(0));
        });

    return newTensorSizeLoop;
}

scf::ForOp applyMaskToOutputTensors(
    FilterOp Op, OpBuilder& builder,
    Location loc, arith::ConstantIndexOp lowerBound,
    arith::ConstantIndexOp upperBound, arith::ConstantIndexOp step,
    ValueRange initArgs, Value maskedTensor, SmallVector<Value> inputTensors) {
    auto applyMask = scf::ForOp::create(
        builder, loc,
        lowerBound, upperBound,
        step, initArgs,
        [&](OpBuilder& nestedBuilder, Location nestedLoc, Value iter, ValueRange nestedblockArgs) {
            // maskVal = tensor.extract %maskedTensor[%iter]
            // if maskVal then
            //     tensorVal =  tensor.extract %inputTensor[%iter]
            //     newTensor = tensor.insert %newTensor[%index], tensorVal
            //     updatedIndex = arith.addi index + step
            //     scf.yield nestedblockArgs + updatedIndex
            // else
            //     scf.yield nestedblockArgs
            auto maskVal = tensor::ExtractOp::create(
                nestedBuilder, nestedLoc, maskedTensor, iter);
            auto ifOp = scf::IfOp::create(
                nestedBuilder, nestedLoc,
                nestedblockArgs.getTypes(), maskVal, true, true);

            // Then Block
            {
                OpBuilder::InsertionGuard insertionGuard(nestedBuilder);
                nestedBuilder.setInsertionPointToStart(ifOp.thenBlock());

                SmallVector<Value> innerIfOpResults{};
                innerIfOpResults.reserve(inputTensors.size());
                for (size_t i{}; i < inputTensors.size(); ++i) {
                    auto tensorVal = tensor::ExtractOp::create(
                        nestedBuilder, nestedLoc, inputTensors[i], iter);

                    auto newTensor = tensor::InsertOp::create(
                        nestedBuilder, nestedLoc,
                        tensorVal, nestedblockArgs[i], nestedblockArgs.back());

                    innerIfOpResults.push_back(newTensor.getResult());
                }

                auto updatedIndex = arith::AddIOp::create(
                    nestedBuilder, nestedLoc,
                    step.getType(), nestedblockArgs.back(), step);

                innerIfOpResults.push_back(updatedIndex.getResult());

                scf::YieldOp::create(nestedBuilder, nestedLoc, innerIfOpResults);
            }

            // Else Block
            {
                OpBuilder::InsertionGuard insertionGuard(nestedBuilder);
                nestedBuilder.setInsertionPointToStart(ifOp.elseBlock());
                scf::YieldOp::create(nestedBuilder, nestedLoc, nestedblockArgs);
            }
            scf::YieldOp::create(nestedBuilder, nestedLoc, ifOp.getResults());
        });

    return applyMask;
}

class FilterOpLowering : public OpConversionPattern<FilterOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        // Create Mask Op
        auto inputTensors = adaptor.getOperands().front();
        auto maskOp = genericMaskedTensorOp(Op, rewriter, Op.getLoc(), inputTensors);

        // Get the size of new tensors created from maskedTensor
        auto nrows = Op.getTable().getType().getNrows();
        auto maskedTensor = maskOp.getResult(0);
        auto lowerBound = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 0);
        auto upperBound = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), nrows);
        auto step = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 1);
        auto newTensorSizeLoop = scfNewTensorSizeLoopFromMask(
            Op, rewriter, Op.getLoc(),
            maskedTensor, lowerBound, upperBound, step);

        // Create new empty tensor from newTensorSizeLoop result
        auto numInputTensors = inputTensors.size();
        auto columns = Op.getTable().getType().getColumns();
        SmallVector<Value> newEmptyTensors{};
        newEmptyTensors.reserve(numInputTensors);
        for (size_t i{}; i < numInputTensors; ++i) {
            auto columnType = columns[i].getDtype();
            auto emptyTensor = tensor::EmptyOp::create(
                rewriter, Op.getLoc(),
                {newTensorSizeLoop.getResult(0)},
                columnType);

            newEmptyTensors.push_back(emptyTensor);
        }

        // Index for newly created tensors
        auto newIndex = arith::ConstantIndexOp::create(rewriter, Op.getLoc(), 0);
        newEmptyTensors.push_back(newIndex);
        // Apply the SCF.ForOp to write values to newEmptyTensors
        ValueRange initArgs{newEmptyTensors};

        auto applyMask = applyMaskToOutputTensors(
            Op, rewriter, Op.getLoc(),
            lowerBound, upperBound, step,
            initArgs, maskedTensor, inputTensors);

        SmallVector<ValueRange> result = {applyMask.getResults().drop_back()};
        rewriter.replaceOpWithMultiple(Op, result);
        return success();
    }
};

class FilterYieldOpLowering : public OpConversionPattern<FilterYieldOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FilterYieldOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<linalg::YieldOp>(
            Op, adaptor.getOperands().back());
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
                                  ConversionPatternRewriter& rewriter) const override {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];

        auto newPred = translatePredicate(Op.getPredicate());
        auto compare = arith::CmpIOp::create(
            rewriter, Op.getLoc(),
            rewriter.getI1Type(), newPred, lhs[0], rhs[0]);

        rewriter.replaceOp(Op, compare.getResult());
        return success();
    }
};

class LogicalAndOpLowering : public OpConversionPattern<LogicalAndOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LogicalAndOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];

        auto arithAnd = arith::AndIOp::create(rewriter, Op.getLoc(), lhs[0], rhs[0]);

        rewriter.replaceOp(Op, arithAnd.getResult());
        return success();
    }
};

class LogicalOrOpLowering : public OpConversionPattern<LogicalOrOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LogicalOrOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto lhs = adaptor.getOperands()[0];
        auto rhs = adaptor.getOperands()[1];

        auto arithOr = arith::OrIOp::create(rewriter, Op.getLoc(), lhs[0], rhs[0]);

        rewriter.replaceOp(Op, arithOr.getResult());
        return success();
    }
};

class LogicalNotOpLowering : public OpConversionPattern<LogicalNotOp> {
    public:
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(LogicalNotOp Op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto rhs = adaptor.getOperands();
        auto i1Type = rewriter.getI1Type();
        auto flipMask = arith::ConstantOp::create(
            rewriter, Op.getLoc(),
            i1Type, rewriter.getIntegerAttr(i1Type, true));

        auto flipVal = arith::XOrIOp::create(
            rewriter, Op.getLoc(),
            i1Type, rhs[0], flipMask.getResult());

        rewriter.replaceOp(Op, flipVal.getResult());
        return success();
    }
};

class OutputOpLowering : public OpConversionPattern<OutputOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(OutputOp Op, OneToNOpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        auto tableType = Op.getTable().getType();
        auto selectAttr = Op.getSelectAttr();
        auto tableColumns = tableType.getColumns();
        auto inputTensors = adaptor.getOperands().front();

        llvm::DenseSet<StringRef> selectedNames;
        for (auto select : selectAttr) {
            auto colName = cast<StringAttr>(select).getValue();
            selectedNames.insert(colName);
        }

        SmallVector<Value> selected;
        for (auto [idx, val] : llvm::enumerate(inputTensors)) {
            if (selectedNames.contains(tableColumns[idx].getName())) {
                selected.push_back(val);
            }
        }

        rewriter.replaceOpWithMultiple(Op, {ValueRange(selected)});
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
        patterns.add<ScanOpLowering>(converter, ctx);
        patterns.add<CmpIOpLowering,
                     LogicalAndOpLowering,
                     LogicalOrOpLowering,
                     LogicalNotOpLowering>(converter, ctx);
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
