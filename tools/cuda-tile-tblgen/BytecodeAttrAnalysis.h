//===- BytecodeAttrAnalysis.h -----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrAnalysis.h - Attribute Bytecode Analysis
//
// This file defines data structures and analysis functions for parsing
// TableGen attribute definitions into intermediate representations suitable
// for bytecode generation and validation.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/TableGen/Record.h"

#include <string>

namespace mlir::tblgen {

/// Represents an attribute with bytecode tag assignment and version info.
/// Version is looked up from CudaTileAttrAlias (for MLIR builtins) or
/// CudaTileAttrDef (for CudaTile-specific attrs) in AttrDefs.td.
struct BytecodeAttr {
  std::string attrName;
  unsigned tagValue;
  std::string sinceVersion;
  bool skipsVersionCheck = false;
};

/// Represents a single enum case with its version info.
struct EnumCase {
  std::string name;
  int value;
  std::string sinceVersion;
};

/// Represents a CudaTile enum attribute for bytecode serialization.
struct CudaTileEnumAttr {
  std::string enumName;
  std::string attrName;
  std::string sinceVersion;
  bool skipsVersionCheck = false;
  llvm::SmallVector<EnumCase, 4> cases;
};

/// Analyzed bytecode attribute structure for code generation.
/// Used by cuda-tile-tblgen to generate:
///   - AttributeTag enum (from bytecodeAttrs, sorted by tagValue)
///   - Version check functions (from bytecodeAttrs, enumAttrs)
///   - Type traits for enum attrs (from enumAttrs)
/// Fields:
///   bytecodeAttrs: Attrs with bytecode tag assignments, sorted by tagValue.
///   enumAttrs: Enum attrs for type trait and version check generation.
struct BytecodeAttrStructure {
  llvm::SmallVector<BytecodeAttr, 0> bytecodeAttrs;
  llvm::SmallVector<CudaTileEnumAttr, 0> enumAttrs;
};

/// Parse and analyze bytecode attribute information from TableGen records.
/// Collects attribute tag assignments from BytecodeAttrOpcodes.td and
/// enum definitions from AttrDefs.td for bytecode code generation.
BytecodeAttrStructure analyzeBytecodeAttrs(const llvm::RecordKeeper &records);

/// Validate that all CudaTileAttrDef attributes have BytecodeAttrTag
/// assignments. Emits fatal errors for missing tags.
mlir::LogicalResult
validateAttrTagAssignments(const llvm::RecordKeeper &records,
                           const BytecodeAttrStructure &structure);

} // namespace mlir::tblgen

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_ANALYSIS_H_
