//===- MinVersionAnalyzer.cpp - Minimum Bytecode Version Analysis -*- C++ -*-=//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MinVersionAnalyzer which determines the minimum
// bytecode version required for serializing a given CUDA Tile IR module.
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Analysis/MinVersionAnalyzer.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

// Include generated attribute, enum, and parameter version lookup. Must precede
// the type version map below, which calls detail::getAttrMinVersionRecursive
// (emitted here) to score attributes embedded in a type parameter.
#define GEN_ATTR_VERSION_MAP
#define GEN_ENUM_VALUE_VERSION_MAP
#define GEN_ATTR_PARAM_VERSION_MAP
#include "AttrBytecode.inc"

// Include generated type version lookup. Its getTypeMinVersionRecursive walker
// calls detail::getAttrMinVersionRecursive for attribute-valued type params.
#define GEN_TYPE_VERSION_MAP
#include "TypeBytecode.inc"

// Include generated per-operation version checkers.
#define GEN_OP_VERSION_CHECKERS
#include "Bytecode.inc"

using namespace mlir;
using namespace mlir::cuda_tile;

FailureOr<BytecodeVersion>
mlir::cuda_tile::getMinBytecodeVersion(ModuleOp module) {
  BytecodeVersion maxVersion = BytecodeVersion::kMinSupportedVersion;
  DenseSet<Type> processedTypes;

  // Callback for type version checking (with deduplication).
  // If we've already processed this type, its version contribution has been
  // recorded in maxVersion, so return minimum to avoid redundant work.
  auto getTypeVersion = [&](Type type) -> BytecodeVersion {
    if (!processedTypes.insert(type).second) {
      return BytecodeVersion::kMinSupportedVersion;
    }
    auto version = detail::getTypeMinVersionRecursive(type);
    return version.value_or(BytecodeVersion::kMinSupportedVersion);
  };

  // Callback for attribute version checking (attr version, enum case, params).
  auto getAttrVersion = [](Attribute attr) -> BytecodeVersion {
    return detail::getAttrMinVersionRecursive(attr).value_or(
        BytecodeVersion::kMinSupportedVersion);
  };

  // Process each operation using generated per-op version checkers.
  module->walk([&](Operation *op) {
#define GEN_OP_VERSION_CHECK_DISPATCH
#include "Bytecode.inc"
    maxVersion = std::max(maxVersion, opVersion);
  });

  return maxVersion;
}
