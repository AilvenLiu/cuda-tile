//===- MinVersionAnalyzer.h - Minimum Bytecode Version Analysis -*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the getMinBytecodeVersion function, which analyzes CUDA
// Tile IR to determine the minimum bytecode version required for serialization.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_BYTECODE_ANALYSIS_MIN_VERSION_ANALYZER_H
#define CUDA_TILE_BYTECODE_ANALYSIS_MIN_VERSION_ANALYZER_H

#include "mlir/Support/LogicalResult.h"

#include "cuda_tile/Bytecode/Common/Version.h"

namespace mlir::cuda_tile {

class ModuleOp;

/// Analyzes CUDA Tile IR to determine the minimum bytecode version required.
/// Walks the IR and inspects operations, types, enum values, and attributes
/// to determine the earliest bytecode version that supports the module.
FailureOr<BytecodeVersion> getMinBytecodeVersion(ModuleOp module);

} // namespace mlir::cuda_tile

#endif // CUDA_TILE_BYTECODE_ANALYSIS_MIN_VERSION_ANALYZER_H
