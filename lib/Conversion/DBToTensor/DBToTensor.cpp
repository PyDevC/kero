// lib/Conversion/DBToTensor/DBToTensor.cpp
//
// DBToTensorPass — lowers the db-dialect to tensor / linalg / arith.
//
// Pass pipeline stage 1 (Developer Guide §5.2).
// After this pass, no db-dialect ops remain in the module.
//
// Lowering table (Developer Guide §5.2, "How Each Op is Lowered"):
//   db.scan    → direct reference to the (converted) function argument
//   db.filter  → two linalg.generic ops (mask + select)
//   db.getcol  → tensor.extract with linalg.index 1 as the row position
//   db.return  → linalg.yield inside the mask linalg.generic body
//   db.project → identity passthrough (elided)
//
// Tensor layout convention (Developer Guide §5.2):
//   !db.table<"t">        → tensor<CxNxf32>   (C=cols static, N=rows dynamic)
//   !db.column<"t","c",T> → tensor<1xNxT>     (N=rows dynamic)
//   !db.result            → tensor<CxNxf32>   (same as table)
//   !db.row               → index             (loop induction variable)

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

// ===========================================================================
// DBToTensorTypeConverter
//
// Maps db-dialect types to standard MLIR tensor/index types.
//
// Mapping (Developer Guide §5.2 "Tensor Layout Convention"):
//   !db.table<"t">        → tensor<?x?xf32>   (both dims dynamic)
//   !db.column<"t","c",T> → tensor<1x?xT>     (T from the ColumnType dtype)
//   !db.result            → tensor<?x?xf32>
//   !db.row               → index
//
// NOTE: Source materialisation is intentionally left as the default
// (nullptr / identity) because there is no valid db-dialect op to insert
// when converting *out* of db types — the conversion is always one-way
// and the ConversionPatternRewriter handles all value replacements.
// ===========================================================================

class DBToTensorTypeConverter : public TypeConverter {
    public:
    explicit DBToTensorTypeConverter(MLIRContext* ctx) : context(ctx) {
        // Keep non-db types as-is.
        addConversion([](Type type) { return type; });

        // !db.result → tensor<?x?xf32>
        addConversion([this](db::ResultType) -> Type {
            return RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                         Float32Type::get(context));
        });

        // !db.table<"t"> → tensor<?x?xf32>
        // The static column count is NOT known here — it comes from the
        // runtime Arrow Table.  Keep both dims dynamic for generality.
        // The actual column count is communicated to the Executor via
        // CompiledQuery.n_cols (Developer Guide §5.6).
        addConversion([this](db::TableType) -> Type {
            return RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                         Float32Type::get(context));
        });

        // !db.column<"t","c",T> → tensor<1x?xT>
        // First dimension is always 1 (single column slice).
        // Second dimension is dynamic (row count known only at runtime).
        addConversion([this](db::ColumnType colType) -> Type {
            return RankedTensorType::get({1, ShapedType::kDynamic},
                                         colType.getDtype());
        });

        // !db.row → index  (the loop induction variable inside filter)
        addConversion([this](db::RowType) -> Type {
            return IndexType::get(context);
        });
    }

    private:
    MLIRContext* context;
};

// ===========================================================================
// ConvertDBScan
//
// db.scan %table : !db.table<"t"> -> !db.result
//   →  (elided; the table tensor becomes the result directly)
//
// The table function argument is already converted to tensor<?x?xf32> by
// the type converter.  db.scan simply passes it through.
// ===========================================================================

class ConvertDBScan : public OpConversionPattern<ScanOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        // Replace scan result with the (already-converted) table argument.
        rewriter.replaceOp(op, adaptor.getTable());
        return success();
    }
};

// ===========================================================================
// ConvertDBReturn
//
// db.return %val : T  →  func.return %val : T
//
// db.return is used as the terminator of filter regions.  After the filter
// lowering the regions are inlined into linalg bodies where the terminator
// becomes linalg.yield (handled inside ConvertDBFilter).  Any db.return
// that still exists at function scope is converted to func.return.
// ===========================================================================

class ConvertDBReturn : public OpConversionPattern<ReturnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

// ===========================================================================
// ConvertDBFilter
//
// db.filter %input { ^bb0(%row : !db.row): ... db.return %cond : i1 }
//   →  Step 1: linalg.generic computing boolean mask tensor<?x?xi1>
//      Step 2: linalg.fill + linalg.generic applying arith.select
//
// Developer Guide §5.2: "db.filter → two linalg.generic"
//
// FIX (session2 §"Issue Found"): The original code called
// linalg::GenericOp::create() and then .getResult(0) on an Operation*.
// The correct pattern is rewriter.create<linalg::GenericOp>(...) which
// returns the typed op, and then .getResult(0) on that op.
//
// FIX: tensor::EmptyOp with dynamic shapes requires a ValueRange of
// dynamic-size SSA values — static shape arrays alone are not accepted
// when any dimension is ShapedType::kDynamic.
// ===========================================================================

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

        // ---- Step 1: compute boolean mask -----------------------------------
        Value mask = generateMaskTensor(inputTensor, op.getRegion(), rewriter, loc);

        // ---- Step 2: apply mask (arith.select) ------------------------------
        // 2a. Allocate zero-initialised output tensor of same shape as input.
        Value emptyOut = buildDynamicEmpty(rewriter, loc, tensorType, inputTensor);

        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(tensorType.getElementType()));
        Value zeroFilled =
            linalg::FillOp::create(rewriter, loc,
                                   ValueRange{zero}, ValueRange{emptyOut})
                .getResult(0);

        // 2b. linalg.generic: for each element select(mask, input, zero).
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
    // -----------------------------------------------------------------------
    // generateMaskTensor
    //
    // Builds a tensor<?x?xi1> by inlining the filter region body.
    // The region's %row argument (type index after conversion) is mapped to
    // linalg.index 1 — the column-major row induction variable.
    //
    // FIX: Use rewriter.create<linalg::GenericOp>() (returns typed op) so
    // that .getResult(0) compiles correctly.
    // -----------------------------------------------------------------------
    Value generateMaskTensor(Value inputTensor, Region& filterRegion,
                             ConversionPatternRewriter& rewriter,
                             Location loc) const {
        auto tensorType = cast<RankedTensorType>(inputTensor.getType());

        // Mask has same shape as input but element type i1.
        auto i1Type = rewriter.getI1Type();
        auto maskType = RankedTensorType::get(tensorType.getShape(), i1Type);

        // Allocate the output mask tensor (dynamic dimensions come from input).
        Value initMask = buildDynamicEmpty(rewriter, loc, maskType, inputTensor);

        // Identity maps: both input and output share the same iteration space.
        auto map = rewriter.getMultiDimIdentityMap(tensorType.getRank());
        SmallVector<AffineMap> maps = {map, map};
        SmallVector<utils::IteratorType> iterators(
            tensorType.getRank(), utils::IteratorType::parallel);

        // FIX: use rewriter.create<> so the return type is linalg::GenericOp
        // and .getResult(0) is valid.
        auto maskOp = rewriter.create<linalg::GenericOp>(
            loc, maskType,
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{initMask},
            maps, iterators,
            [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
                // Map the !db.row block argument to linalg.index 1.
                // (Dimension 0 = columns, dimension 1 = rows — column-major layout.)
                Value rowIV =
                    nestedBuilder.create<linalg::IndexOp>(nestedLoc, 1);

                IRMapping mapper;
                mapper.map(filterRegion.getArgument(0), rowIV);

                // Clone all ops from the region body except the terminator.
                for (auto& nestedOp : filterRegion.front().without_terminator())
                    nestedBuilder.clone(nestedOp, mapper);

                // The terminator is db.return %cond : i1.
                // Emit linalg.yield with the mapped predicate value.
                auto returnOp =
                    cast<db::ReturnOp>(filterRegion.front().getTerminator());
                Value result = mapper.lookupOrDefault(returnOp.getInput());
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, result);
            });

        return maskOp.getResult(0);
    }

    // -----------------------------------------------------------------------
    // buildDynamicEmpty
    //
    // Creates a tensor.empty() whose dynamic dimensions are extracted from
    // a reference tensor via tensor.dim ops.
    //
    // FIX: tensor::EmptyOp::create() requires explicit Value operands for
    // every dynamic dimension (ShapedType::kDynamic).  Passing a static
    // shape array with kDynamic entries without the corresponding Value
    // operands is invalid and triggers the assertion seen in session2.
    // -----------------------------------------------------------------------
    static Value buildDynamicEmpty(OpBuilder& builder, Location loc,
                                   RankedTensorType targetType,
                                   Value referenceSource) {
        ArrayRef<int64_t> shape = targetType.getShape();
        SmallVector<Value> dynSizes;

        for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
            if (shape[i] == ShapedType::kDynamic) {
                Value dim = builder.create<tensor::DimOp>(loc, referenceSource,
                                                          builder.create<arith::ConstantIndexOp>(loc, i));
                dynSizes.push_back(dim);
            }
        }

        return tensor::EmptyOp::create(builder, loc, shape,
                                       targetType.getElementType(), dynSizes);
    }
};

// ===========================================================================
// ConvertDBGetCol
//
// db.getcol %row, %col : (!db.row, !db.column<"t","c",T>) -> T
//   →  tensor.extract %col[c0, %row] : tensor<1x?xT>
//
// %row after conversion is an index value (from RowType → IndexType).
// c0 is always 0 (the single-column first dimension).
// ===========================================================================

class ConvertDBGetCol : public OpConversionPattern<GetColumnOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GetColumnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        Location loc = op.getLoc();
        Value rowIndex = adaptor.getRow();
        Value columnTensor = adaptor.getColumn();

        // Ensure rowIndex is index type (RowType → IndexType converter handles
        // this, but guard defensively).
        if (!isa<IndexType>(rowIndex.getType())) {
            rowIndex = arith::IndexCastOp::create(rewriter, loc,
                                                  rewriter.getIndexType(), rowIndex);
        }

        // tensor.extract %col[0, rowIndex]
        Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
        SmallVector<Value> indices{zeroIdx, rowIndex};
        Value extracted =
            tensor::ExtractOp::create(rewriter, loc, columnTensor, indices);

        rewriter.replaceOp(op, extracted);
        return success();
    }
};

// ===========================================================================
// ConvertDBProject
//
// db.project %input : !db.result -> !db.result
//   →  (elided; passes the input tensor through)
//
// Developer Guide §4.2: "db.project — Not yet implemented in
// DBToTensorPass — planned for future work."  For now project is a no-op
// passthrough at the tensor level.
// ===========================================================================

class ConvertDBProject : public OpConversionPattern<ProjectOp> {
    public:
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(ProjectOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter& rewriter) const override {
        rewriter.replaceOp(op, adaptor.getFilter());
        return success();
    }
};

// ===========================================================================
// DBToTensor pass body
// ===========================================================================

struct DBToTensor : impl::DBToTensorBase<DBToTensor> {
    using DBToTensorBase::DBToTensorBase;

    void runOnOperation() override {
        MLIRContext* context = &getContext();
        DBToTensorTypeConverter typeConverter(context);
        auto* module = getOperation();

        // Legal target: tensor/linalg/arith are fully legal.
        // All db-dialect ops are illegal.
        // func.func and func.return are dynamically legal once their
        // types have been converted.
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

        // Convert function signatures and call sites.
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
