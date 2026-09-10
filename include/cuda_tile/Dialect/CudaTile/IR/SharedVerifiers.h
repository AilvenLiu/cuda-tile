//===- SharedVerifiers.h - CUDA Tile Shared Verifiers -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
#define CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/LogicalResult.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "cuda_tile/Dialect/CudaTile/IR/Types.h"

namespace mlir {
namespace cuda_tile {

//===----------------------------------------------------------------------===//
// View Load and Store Utilities
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult verifyOptHintsCommon(Op op) {
  Operation *opInst = op->getOperation();
  auto hintsAttr =
      opInst->getAttrOfType<OptimizationHintsAttr>("optimization_hints");
  if (!hintsAttr || hintsAttr.getValue().empty()) {
    return success();
  }
  if (failed(
          hintsAttr.verifyWithOp(op->getOperation(), hintsAttr.getValue()))) {
    return op->emitOpError("Optimization hints verification failed");
  }
  return success();
}

static inline LogicalResult verifyViewIndexTileMatch(Operation *op,
                                                     Value viewVal,
                                                     ValueRange indexVals,
                                                     Value tileVal) {
  auto viewIface = dyn_cast<TileView>(viewVal.getType());
  if (!viewIface) {
    return op->emitOpError("expected a tile view type for the view operand");
  }
  TypeRange indexTypes = indexVals.getTypes();
  Type tileType = tileVal.getType();

  if (indexTypes.size() != viewIface.getViewIndexRank()) {
    return op->emitOpError()
           << "expected " << viewIface.getViewIndexRank()
           << " index operands (based on view type), got " << indexTypes.size();
  }

  if (tileType != viewIface.getViewTileType()) {
    return op->emitOpError()
           << "expected tile type to be " << viewIface.getViewTileType()
           << " (based on view type), got " << tileType;
  }

  if (failed(viewIface.verifyIndices([&]() { return op->emitOpError(); },
                                     indexTypes))) {
    return failure();
  }

  return success();
}

template <typename LoadStoreOp>
static LogicalResult verifyViewLoadStoreCommon(LoadStoreOp op) {
  if (failed(verifyViewIndexTileMatch(op->getOperation(), op->getView(),
                                      op->getIndex(), op->getTile()))) {
    return failure();
  }
  return verifyOptHintsCommon(op);
}

static inline LogicalResult verifyViewLoadStoreCommon(Operation *op,
                                                      Value viewVal,
                                                      ValueRange indexVals,
                                                      Value tileVal) {
  return verifyViewIndexTileMatch(op, viewVal, indexVals, tileVal);
}

/// Verifies that every dimension in `shape`
///   • is a positive compile‑time constant,
///   • is a power of two, and
///   • the total element count does not exceed `maxTileNumElements`.
static inline LogicalResult
verifyTileSize(function_ref<InFlightDiagnostic()> emitError,
               ArrayRef<int64_t> shape) {
  constexpr int64_t kMaxElems = maxTileNumElements;

  int64_t numElems = 1;
  for (int64_t dim : shape) {
    // Dimension must be positive.
    if (dim <= 0)
      return emitError() << "all dimensions must be positive constants, got "
                         << shape;

    // Dimension must be a power of two.
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(dim)))
      return emitError() << "all dimensions must be powers of two, got "
                         << shape;

    // Guard against overflow before multiplying.
    if (numElems > kMaxElems / dim)
      return emitError() << "tile would exceed the maximum of " << kMaxElems
                         << " elements";

    numElems *= dim;
  }

  return success();
}

template <typename OpTy>
static inline LogicalResult verifyFtz(OpTy op, bool ftz) {
  Type ty = getElementTypeOrSelf(op.getResult().getType());

  // Check flush-to-zero modifier compatibility
  // FTZ: When set, subnormal inputs and results are flushed to sign-preserving
  // zero.
  if (ftz && !ty.isF32())
    return op.emitOpError("flush_to_zero modifier only supported for f32 data "
                          "type, but got: ")
           << ty;
  return success();
}

template <typename OpTy>
static inline LogicalResult verifyApprox(OpTy op, bool approx) {
  Type ty = getElementTypeOrSelf(op.getResult().getType());
  if (approx && !ty.isF32())
    return op.emitOpError(
               "approx modifier only supported for f32 data type, but got: ")
           << ty;
  return success();
}

namespace detail {

template <typename OpTy>
static inline LogicalResult
verifyDivSqrtCommonFPModifiers(OpTy op, bool hasRoundingMode, bool approx,
                               bool full, bool ftz) {
  if (approx && full)
    return op.emitOpError(
        "approx modifier and full modifier are mutually exclusive");

  Type ty = getElementTypeOrSelf(op.getResult().getType());
  if (ftz && !ty.isF32())
    return op.emitOpError("flush_to_zero modifier only supported for f32 data "
                          "type, but got: ")
           << ty;

  if (approx && !ty.isF32())
    return op.emitOpError(
               "approx modifier only supported for f32 data type, but got: ")
           << ty;

  if (full && !ty.isF32())
    return op.emitOpError(
               "full modifier only supported for f32 data type, but got: ")
           << ty;

  if (hasRoundingMode) {
    if (approx)
      return op.emitOpError("rounding mode does not allow approx modifier");
    if (full)
      return op.emitOpError("rounding mode does not allow full modifier");
  }
  return success();
}

} // namespace detail

template <typename OpTy>
static inline LogicalResult verifyDivFPModifiers(OpTy op, bool hasRoundingMode,
                                                 bool approx, bool full,
                                                 bool ftz) {

  return detail::verifyDivSqrtCommonFPModifiers(op, hasRoundingMode, approx,
                                                full, ftz);
}

template <typename OpTy>
static inline LogicalResult verifySqrtFPModifiers(OpTy op, bool hasRoundingMode,
                                                  bool approx, bool full,
                                                  bool ftz) {

  return detail::verifyDivSqrtCommonFPModifiers(op, hasRoundingMode, approx,
                                                full, ftz);
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

template <typename OpTy, typename PointeeType>
static inline LogicalResult verifyAlloca(OpTy op) {
  // Alignment must be power of 2.
  int64_t alignment = op.getAlignment();
  bool isPowerOfTwo = alignment > 0 && ((alignment & (alignment - 1)) == 0);
  if (!isPowerOfTwo)
    return op.emitOpError() << "'alignment' must be power of two";

  auto ptrType =
      cast<PointeeType>(getElementTypeOrSelf(op.getResult().getType()));
  Type pointeeTy = ptrType.getPointeeType();
  int64_t sizeInBytes = 0;
  if (auto intTy = dyn_cast<IntegerType>(pointeeTy)) {
    bool isIntOne = intTy.isInteger(1);
    sizeInBytes = (isIntOne) ? 1 : intTy.getWidth() / 8;
  }
  if (auto floatTy = dyn_cast<FloatType>(pointeeTy))
    sizeInBytes = APFloat::getSizeInBits(floatTy.getFloatSemantics()) / 8;
  if (alignment < sizeInBytes)
    return op.emitOpError()
           << "'alignment' (" << alignment
           << ") must be at least the natural size (" << sizeInBytes
           << " bytes) for element type " << pointeeTy;

  return success();
}

//===----------------------------------------------------------------------===//
// Atomic Operations Utilities
//===----------------------------------------------------------------------===//

/// Verifies RMW mode compatibility with element type for atomic operations.
template <typename OpTy>
static inline LogicalResult verifyAtomicRMWMode(OpTy op, AtomicRMWMode mode,
                                                Type elementType) {
  switch (mode) {
  case AtomicRMWMode::AND:
  case AtomicRMWMode::OR:
  case AtomicRMWMode::XOR:
  case AtomicRMWMode::ADD:
  case AtomicRMWMode::MAX:
  case AtomicRMWMode::MIN:
  case AtomicRMWMode::UMAX:
  case AtomicRMWMode::UMIN: {
    auto integerTy = dyn_cast_or_null<IntegerType>(elementType);
    if (!integerTy || (!integerTy.isInteger(32) && !integerTy.isInteger(64)))
      return op.emitOpError("'") << stringifyAtomicRMWMode(mode)
                                 << "' works only with integers i32 and i64";
    break;
  }
  case AtomicRMWMode::ADDF: {
    auto floatTy = dyn_cast_or_null<FloatType>(elementType);
    if (!floatTy || (!floatTy.isF32() && !floatTy.isF64() && !floatTy.isF16() &&
                     !floatTy.isBF16())) {
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with floats f16, bf16, f32, and f64";
    }
    break;
  }
  case AtomicRMWMode::XCHG: {
    auto integerTy = dyn_cast_or_null<IntegerType>(elementType);
    auto floatTy = dyn_cast_or_null<FloatType>(elementType);
    if (!integerTy && !floatTy)
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with integers or float of 32 or 64 bitwidth";
    int64_t bitwidth = elementType.getIntOrFloatBitWidth();
    if (bitwidth != 32 && bitwidth != 64)
      return op.emitOpError("'")
             << stringifyAtomicRMWMode(mode)
             << "' works only with integers or float of 32 or 64 bitwidth";
  }
  }
  return success();
}

/// Verifies memory ordering semantics for atomic operations.
template <typename OpTy>
static inline LogicalResult
verifyAtomicMemoryOrdering(OpTy op, MemoryOrderingSemantics semantics) {
  if (semantics != MemoryOrderingSemantics::RELAXED &&
      semantics != MemoryOrderingSemantics::ACQUIRE &&
      semantics != MemoryOrderingSemantics::RELEASE &&
      semantics != MemoryOrderingSemantics::ACQ_REL) {
    return op.emitOpError("memory ordering semantics must be one of: "
                          "relaxed, acquire, release, acq_rel");
  }
  return success();
}

/// Extract PtrAttrAttr from a TileView type's embedded TensorViewType.
/// Returns failure if the view type is not recognized.
static inline FailureOr<PtrAttrAttr> getViewPtrAttr(Operation *op,
                                                    Type viewType) {
  if (auto pv = dyn_cast<PartitionViewType>(viewType)) {
    return pv.getTensorView().getPtrAttr();
  }
  if (auto sv = dyn_cast<StridedViewType>(viewType)) {
    return sv.getTensorView().getPtrAttr();
  }
  if (auto gv = dyn_cast<GatherScatterViewType>(viewType)) {
    return gv.getTensorView().getPtrAttr();
  }
  if (auto tv = dyn_cast<TensorViewType>(viewType)) {
    return tv.getPtrAttr();
  }
  return op->emitOpError("unknown TileView type: ") << viewType;
}

/// Verify that a TileView operand does not have a restricted ptr_attr.
template <typename OpT>
static LogicalResult verifyUnrestrictedViewPtrAttr(OpT op, Value viewOperand) {
  auto ptrAttrOrErr = getViewPtrAttr(op.getOperation(), viewOperand.getType());
  if (failed(ptrAttrOrErr)) {
    return failure();
  }
  return success();
}

} // namespace cuda_tile
} // namespace mlir

#endif // CUDA_TILE_DIALECT_CUDATILE_IR_SHAREDVERIFIERS_H
