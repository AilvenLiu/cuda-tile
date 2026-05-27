//===- Attributes.cpp - CUDA Tile Attribute Verifiers -----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "cuda_tile/Dialect/CudaTile/IR/Dialect.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include <optional>

using namespace mlir;
using namespace mlir::cuda_tile;

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Optimization Hints Validation Helpers
//===----------------------------------------------------------------------===//

namespace {
/// Result of validation - contains optional value and error message
template <typename T>
struct ValidationResult {
  std::optional<T> value;
  std::string errorMessage;

  bool isValid() const { return value.has_value(); }
  explicit operator bool() const { return isValid(); }
};
} // namespace

static InFlightDiagnostic
emitDiagnostic(Location loc, StringRef msg = StringRef(),
               DiagnosticSeverity severity = DiagnosticSeverity::Warning) {
  return severity == DiagnosticSeverity::Error ? emitError(loc, msg)
                                               : emitWarning(loc, msg);
}

/// Validate num_cta_in_cga parameter and return the value if valid.
/// Returns nullopt with error message if invalid (when key is provided).
/// Pass empty key from getters to skip error message construction.
static ValidationResult<uint64_t> validateNumCTAInCGA(Attribute attr,
                                                      StringRef context) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  if (!intAttr) {
    return {std::nullopt, ("integer value expected for " + context + "." +
                           OptimizationHintsAttr::kNumCTAInCGA)
                              .str()};
  }
  // Ampere/ada don't support multiple CTAs in a CGA.
  static const SmallVector<StringLiteral, 5> restrictedArchs = {
      "sm_80", "sm_86", "sm_87", "sm_88", "sm_89"};
  bool requiresSingleCTA = any_of(restrictedArchs, [&](StringRef arch) {
    return context.starts_with(arch);
  });
  uint64_t numCTA = intAttr.getInt();
  if (requiresSingleCTA && numCTA != 1) {
    return {std::nullopt, ("expected 1 for " + context + "." +
                           OptimizationHintsAttr::kNumCTAInCGA)
                              .str()};
  }
  // Must be power of 2, non-zero, and <= 16
  if ((numCTA == 0) || (numCTA > 16) || ((numCTA & (numCTA - 1)) != 0)) {
    return {std::nullopt, ("expected power-of-two ≤ 16 for " + context + "." +
                           OptimizationHintsAttr::kNumCTAInCGA)
                              .str()};
  }
  return {numCTA, ""};
}

static ValidationResult<uint64_t>
validateNumWorkerWarpsPerCTA(Attribute attr, StringRef context) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  if (!intAttr) {
    return {std::nullopt, ("integer value expected for " + context + "." +
                           OptimizationHintsAttr::kNumWorkerWarpsPerCTA)
                              .str()};
  }
  int numWarps = intAttr.getInt();
  // Must be power of 2, non-zero, and <= 32
  if ((numWarps == 0) || (numWarps > 32) || ((numWarps & (numWarps - 1)) != 0)) {
    return {std::nullopt, ("expected power-of-two ≤ 32 for " + context + "." +
                           OptimizationHintsAttr::kNumWorkerWarpsPerCTA)
                              .str()};
  }
  // TODO: Currently only support 4 or 8 warps for no functionality check.
  if (numWarps != 4 && numWarps != 8) {
    numWarps = std::clamp(numWarps, 4, 8);
  }
  return {numWarps, ""};
}

/// Validate allow_tma parameter and return the value if valid.
/// Returns nullopt with error message if invalid (when key is provided).
/// Pass empty key from getters to skip error message construction.
static ValidationResult<bool> validateAllowTMA(Attribute attr,
                                               StringRef context) {
  auto boolAttr = dyn_cast_or_null<BoolAttr>(attr);
  if (!boolAttr) {
    return {std::nullopt, ("boolean value expected for " + context + "." +
                           OptimizationHintsAttr::kAllowTMA)
                              .str()};
  }
  return {boolAttr.getValue(), ""};
}

/// Validate latency parameter and return the value if valid.
/// Returns nullopt with error message if invalid (when key is provided).
/// Pass empty key from getters to skip error message construction.
static ValidationResult<int64_t> validateLatency(Attribute attr,
                                                 StringRef context) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  if (!intAttr) {
    return {std::nullopt, ("integer value expected for " + context + "." +
                           OptimizationHintsAttr::kLatency)
                              .str()};
  }

  int64_t val = intAttr.getInt();
  // Must be in range [1, 10]
  if ((val < 1) || (val > 10)) {
    return {std::nullopt,
            ("integer value in the range [1, 10] is expected for " + context +
             "." + OptimizationHintsAttr::kLatency)
                .str()};
  }

  return {val, ""};
}

/// Validate occupancy parameter and return the value if valid.
/// Returns nullopt with error message if invalid (when key is provided).
/// Pass empty key from getters to skip error message construction.
static ValidationResult<int64_t> validateOccupancy(Attribute attr,
                                                   StringRef context) {
  auto intAttr = dyn_cast_or_null<IntegerAttr>(attr);
  if (!intAttr) {
    return {std::nullopt, ("integer value expected for " + context + "." +
                           OptimizationHintsAttr::kOccupancy)
                              .str()};
  }

  int64_t val = intAttr.getInt();
  // Must be in range [1, 32]
  if ((val < 1) || (val > 32)) {
    return {std::nullopt,
            ("integer value in the range [1, 32] is expected for " + context +
             "." + OptimizationHintsAttr::kOccupancy)
                .str()};
  }

  return {val, ""};
}

/// Helper function to retrieve an attribute from SM-specific or default entry.
/// Returns the attribute if found, otherwise std::nullopt.
std::optional<Attribute>
OptimizationHintsAttr::getAttributeForSmOrDefault(DictionaryAttr value,
                                                  StringRef sm, StringRef key) {
  if (value.empty())
    return std::nullopt;

  // Try SM-specific entry first
  if (auto smEntry = value.getAs<DictionaryAttr>(sm))
    if (Attribute attr = smEntry.get(key))
      return attr;

  // Fall back to default entry
  if (auto defaultEntry = value.getAs<DictionaryAttr>(kDefault))
    if (Attribute attr = defaultEntry.get(key))
      return attr;

  return std::nullopt;
}

// Return failure() if hints are not supported for current operations/target
LogicalResult OptimizationHintsAttr::verifyParamWithContext(
    Location loc, StringRef context, ArrayRef<StringRef> keysValidForOperation,
    DictionaryAttr &attr) {
  // Fast return if not warning about hints
  if (!cast<CudaTileDialect>(getDialect()).getWarnUnsupportedHints())
    return success();

  // Handle unallowed architecture key
  if (!isKnownKey(context))
    return emitDiagnostic(loc) << "unknown hint key " << context;

  LogicalResult res = success();
  for (auto param : attr) {
    StringRef key = param.getName().strref();

    if (!keysValidForOperation.empty() &&
        !is_contained(keysValidForOperation, key)) {
      res = emitDiagnostic(loc)
            << key << " is not known hint for current Operation";
      continue;
    }

    if (key == kNumCTAInCGA) {
      auto result = validateNumCTAInCGA(param.getValue(), context);
      if (!result.isValid())
        res = emitDiagnostic(loc) << result.errorMessage;
    } else if (key == kNumWorkerWarpsPerCTA) {
      auto result = validateNumWorkerWarpsPerCTA(param.getValue(), context);
      if (!result.isValid())
        res = emitDiagnostic(loc) << result.errorMessage;
    } else if (key == kAllowTMA) {
      auto result = validateAllowTMA(param.getValue(), context);
      if (!result.isValid())
        res = emitDiagnostic(loc) << result.errorMessage;
    } else if (key == kLatency) {
      auto result = validateLatency(param.getValue(), context);
      if (!result.isValid())
        res = emitDiagnostic(loc) << result.errorMessage;
    } else if (key == kOccupancy) {
      auto result = validateOccupancy(param.getValue(), context);
      if (!result.isValid())
        res = emitDiagnostic(loc) << result.errorMessage;
    } else {
      // Unknown parameter
      res = emitDiagnostic(loc)
            << "unknown param " << key << " for " << context;
    }
  }

  // Return failure if warnings were printed
  // Error (if errorOnHints not set) would be suppressed outside
  return res;
}

LogicalResult
OptimizationHintsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              DictionaryAttr value) {
  for (NamedAttribute entry : value.getValue()) {
    if (!isa<DictionaryAttr>(entry.getValue()))
      return emitError()
             << "expected dictionary attribute for optimization_hints entry `"
             << entry.getName().strref() << "` got value=" << entry.getValue();
  }
  return success();
}

LogicalResult OptimizationHintsAttr::verifyWithOp(Operation *op,
                                                  DictionaryAttr value) {
  Location loc = op->getLoc();
  bool errorOnHints = cast<CudaTileDialect>(getDialect()).getErrorOnHints();
  SmallVector<StringRef, 4> keysValidForOperation;
  if (op != nullptr) {
    // Initialize list of supported hints for EntryOp
    if (isa<EntryOp>(op)) {
      keysValidForOperation.push_back(kNumCTAInCGA);
      keysValidForOperation.push_back(kNumWorkerWarpsPerCTA);
      keysValidForOperation.push_back(kOccupancy);
    }
    // Initialize list of supported hints for Load/Store Ops
    if (isa<LoadViewTkoOp, StoreViewTkoOp, LoadPtrTkoOp, StorePtrTkoOp>(op)) {
      keysValidForOperation.push_back(kLatency);
      if (isa<LoadViewTkoOp, StoreViewTkoOp>(op))
        keysValidForOperation.push_back(kAllowTMA);
    }
  }

  for (NamedAttribute entry : value.getValue()) {
    StringRef key = entry.getName().strref();

    auto innerDict = dyn_cast<DictionaryAttr>(entry.getValue());
    if (!innerDict)
      return op->emitOpError()
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << entry.getValue();

    if (failed(
            verifyParamWithContext(loc, key, keysValidForOperation, innerDict)))
      if (errorOnHints)
        return failure();
  }

  return success();
}

std::optional<int> OptimizationHintsAttr::getNumCTAInCGA(StringRef sm) {
  auto attrOpt = getAttributeForSmOrDefault(getValue(), sm, kNumCTAInCGA);
  if (!attrOpt)
    return std::nullopt;

  // Validate without constructing error message (key not passed)
  auto result = validateNumCTAInCGA(*attrOpt, sm);
  if (result.isValid())
    return static_cast<int>(*result.value);

  return std::nullopt;
}

std::optional<int>
OptimizationHintsAttr::getNumWorkerWarpsPerCTA(StringRef sm) {
  auto attrOpt =
      getAttributeForSmOrDefault(getValue(), sm, kNumWorkerWarpsPerCTA);
  if (!attrOpt)
    return std::nullopt;

  // Validate without constructing error message (key not passed)
  auto result = validateNumWorkerWarpsPerCTA(*attrOpt, sm);
  if (result.isValid())
    return static_cast<int>(*result.value);

  return std::nullopt;
}

std::optional<bool> OptimizationHintsAttr::getAllowTMA(StringRef sm) {
  auto attrOpt = getAttributeForSmOrDefault(getValue(), sm, kAllowTMA);
  if (!attrOpt)
    return std::nullopt;

  // Validate without constructing error message (key not passed)
  auto result = validateAllowTMA(*attrOpt, sm);
  return result.value;
}

std::optional<int> OptimizationHintsAttr::getLatency(StringRef sm) {
  auto attrOpt = getAttributeForSmOrDefault(getValue(), sm, kLatency);
  if (!attrOpt)
    return std::nullopt;

  // Validate without constructing error message (key not passed)
  auto result = validateLatency(*attrOpt, sm);
  if (result.isValid())
    return static_cast<int>(*result.value);

  return std::nullopt;
}

std::optional<int> OptimizationHintsAttr::getOccupancy(StringRef sm) {
  auto attrOpt = getAttributeForSmOrDefault(getValue(), sm, kOccupancy);
  if (!attrOpt)
    return std::nullopt;

  // Validate without constructing error message (key not passed)
  auto result = validateOccupancy(*attrOpt, sm);
  if (result.isValid())
    return static_cast<int>(*result.value);

  return std::nullopt;
}

Attribute OptimizationHintsAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};
  if (succeeded(parser.parseOptionalGreater()))
    return OptimizationHintsAttr::get(parser.getContext(),
                                      DictionaryAttr::get(parser.getContext()));

  NamedAttrList entries;

  auto parseOneEntry = [&]() -> ParseResult {
    std::string key;
    Attribute rawAttr;
    DictionaryAttr dataDict;
    if (parser.parseKeywordOrString(&key) || parser.parseEqual() ||
        parser.parseAttribute(rawAttr))
      return failure();

    if (entries.get(key))
      return parser.emitError(parser.getCurrentLocation())
             << "duplicate optimization_hints key `" << key << "`";

    dataDict = dyn_cast<DictionaryAttr>(rawAttr);
    if (!dataDict)
      return parser.emitError(parser.getCurrentLocation())
             << "expected dictionary attribute for optimization_hints entry `"
             << key << "` got value=" << rawAttr;

    entries.append(key, dataDict);
    return success();
  };
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseOneEntry))
    return {};
  if (parser.parseGreater())
    return {};

  return OptimizationHintsAttr::get(
      parser.getContext(), parser.getBuilder().getDictionaryAttr(entries));
}

void OptimizationHintsAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleaveComma(getValue(), printer, [&](NamedAttribute attr) {
    printer << attr.getName().strref() << " = {";
    llvm::interleaveComma(mlir::cast<DictionaryAttr>(attr.getValue()), printer,
                          [&](NamedAttribute na) {
                            printer << na.getName().strref() << " = ";
                            printer.printAttributeWithoutType(na.getValue());
                          });
    printer << "}";
  });
  printer << ">";
}

LogicalResult DivByAttr::verifyWithAssumeOp(Operation *op) const {
  auto assumeOp = llvm::cast<AssumeOp>(op);

  // Make sure divisor is a positive power of 2.
  uint64_t divisor = getDivisor();
  bool isPowerOfTwo = divisor > 0 && ((divisor & (divisor - 1)) == 0);
  if (!isPowerOfTwo)
    return op->emitOpError() << "'" << name << "' divisor must be a power of 2";

  if (!llvm::all_equal({getEvery().has_value(), getAlong().has_value()}))
    return op->emitOpError()
           << "'" << name << "' 'every'/'along' must be used in combination";

  // Verify that the divisor is not larger than 4611686018427387904. This is a
  // technical limitation of the current implementation that could be lifted.
  if (divisor > 4611686018427387904)
    return op->emitOpError() << "'" << name << "' divisor is too large";

  // TensorViewType
  if (auto tensorViewType =
          llvm::dyn_cast<cuda_tile::TensorViewType>(assumeOp.getType())) {
    if (getEvery().has_value())
      return op->emitOpError() << "'" << name
                               << "' 'every'/'along' cannot be used if the "
                                  "constrained value is a tensor_view";
    return success();
  }

  // TileType
  auto tileType = llvm::dyn_cast<cuda_tile::TileType>(assumeOp.getType());
  if (!tileType)
    return op->emitOpError() << "'" << name
                             << "' is valid only for tile of integer/pointer "
                                "or tensor_view values";
  if (tileType.getRank() == 0 && getEvery().has_value())
    return op->emitOpError() << "'" << name
                             << "' 'every'/'along' cannot be used if the "
                                "constrained value is a 0D tile";
  Type elType = tileType.getElementType();
  if (!llvm::isa<cuda_tile::PointerType, IntegerType>(elType))
    return op->emitOpError() << "'" << name
                             << "' is valid only for tile of integer/pointer "
                                "or tensor_view values";

  // Verify every/along.
  if (!getEvery().has_value())
    return success();
  if (*getAlong() < 0 || *getAlong() >= tileType.getRank())
    return op->emitOpError()
           << "'" << name << "' every_dim (" << *getAlong()
           << ") must be >= 0 and < tile rank (" << tileType.getRank() << ")";
  if (*getEvery() < 0 || *getEvery() > tileType.getDimSize(*getAlong()))
    return op->emitOpError() << "expected '" << name
                             << "' every_dim to be within 0 and the size of "
                                "the respective dimension ("
                             << tileType.getDimSize(*getAlong()) << ")";
  return success();
}

Attribute DivByAttr::parse(AsmParser &parser, Type odsType) {
  // Parse literal '<'.
  if (parser.parseLess())
    return {};

  // Parse variable 'divisor'.
  uint64_t divisor = 0;
  if (parser.parseInteger(divisor)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse parameter 'divisor' which is expected to "
                     "be an integer");
    return {};
  }

  // Parse 'every' and 'along'.
  std::optional<int64_t> every = std::nullopt;
  std::optional<int64_t> along = std::nullopt;
  if (succeeded(parser.parseOptionalComma())) {
    // Parse optional every/along.
    int64_t everyVal = -1, alongVal = -1;
    if (parser.parseKeyword("every") || parser.parseInteger(everyVal) ||
        parser.parseKeyword("along") || parser.parseInteger(alongVal))
      return {};
    every = everyVal;
    along = alongVal;
  }

  // Parse literal '>'.
  if (parser.parseGreater())
    return {};

  return DivByAttr::get(parser.getContext(), divisor, every, along);
}

void DivByAttr::print(AsmPrinter &printer) const {
  printer << "<" << getDivisor();
  if (getEvery().has_value())
    printer << ", every " << *getEvery() << " along " << *getAlong();
  printer << ">";
}

LogicalResult SameElementsAttr::verifyWithAssumeOp(Operation *op) const {
  auto assumeOp = llvm::cast<AssumeOp>(op);
  auto tileType = llvm::dyn_cast<cuda_tile::TileType>(assumeOp.getType());
  if (!tileType)
    return op->emitOpError()
           << "'" << name
           << "' is valid only for tile of integer/pointer values";
  if (!llvm::isa<cuda_tile::PointerType, IntegerType>(
          tileType.getElementType()))
    return op->emitOpError()
           << "'" << name
           << "' is valid only for tile of integer/pointer values";
  if (getValues().size() != tileType.getRank())
    return op->emitOpError()
           << "expected number of values in '" << name << "' ("
           << getValues().size() << ") to match rank of constrained tile ("
           << tileType.getRank() << ")";
  for (int64_t i = 0, e = tileType.getRank(); i < e; ++i) {
    if (getValues()[i] < 0 || getValues()[i] > tileType.getDimSize(i))
      return op->emitOpError()
             << "expected '" << name << "' value " << i
             << " to be within 0 and the size of the respective dimension ("
             << tileType.getDimSize(i) << ")";
  }
  return success();
}

LogicalResult BoundedAttr::verifyWithAssumeOp(Operation *op) const {
  auto tileType =
      llvm::dyn_cast<cuda_tile::TileType>(llvm::cast<AssumeOp>(op).getType());
  if (!tileType)
    return op->emitOpError()
           << "'" << name << "' is valid only for tile of integer values";
  auto intType = llvm::dyn_cast<IntegerType>(tileType.getElementType());
  if (!intType)
    return op->emitOpError()
           << "'" << name << "' is valid only for tile of integer values";
  int64_t minVal = getMinSignedValueForBitwidth(intType.getWidth());
  int64_t maxVal = getMaxSignedValueForBitwidth(intType.getWidth());
  if (getLb().has_value() && (*getLb() > maxVal || *getLb() < minVal))
    return op->emitOpError()
           << "'" << name << "' expects lower bound to be within [" << minVal
           << ", " << maxVal << "]";
  if (getUb().has_value() && (*getUb() > maxVal || *getUb() < minVal))
    return op->emitOpError()
           << "'" << name << "' expects upper bound to be within [" << minVal
           << ", " << maxVal << "]";
  if (getLb().has_value() && getUb().has_value() && *getLb() > *getUb())
    return op->emitOpError()
           << "'" << name
           << "' expects lower bound to be less than or equal to upper bound";
  return success();
}

//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILexicalBlockAttr,
                   DISubprogramAttr>(attr);
}

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DIFileAttr, DILocalScopeAttr>(attr);
}

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DISubprogramAttr>(attr);
}

void CudaTileDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc"
      >();
}
