//===- BytecodeAttrCodeGen.h ------------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrCodeGen.h - Attribute Bytecode Code Generation
//
// This file declares functions for generating C++ code from analyzed
// attribute bytecode structures.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_

#include "llvm/Support/raw_ostream.h"

namespace mlir::tblgen {

struct BytecodeAttrStructure;

//===----------------------------------------------------------------------===//
// Code Generation Entry Points
//===----------------------------------------------------------------------===//

/// Generate the AttributeTag enum.
void generateAttrTagEnum(const BytecodeAttrStructure &structure,
                         llvm::raw_ostream &os);

/// Generate runtime version checking function for attribute tags.
void generateAttrVersionCheck(const BytecodeAttrStructure &structure,
                              llvm::raw_ostream &os);

/// Generate is_cuda_tile_enum type trait for BytecodeWriter.
void generateEnumTypeTrait(const BytecodeAttrStructure &structure,
                           llvm::raw_ostream &os);

/// Generate is_cuda_tile_enum_attr type trait and symbolizeEnum for
/// BytecodeReader.
void generateEnumAttrTypeTrait(const BytecodeAttrStructure &structure,
                               llvm::raw_ostream &os);

/// Generate runtime version checking for enum attributes.
void generateEnumAttrVersionCheck(const BytecodeAttrStructure &structure,
                                  llvm::raw_ostream &os);

/// Generate per-value version checking for enum values.
void generateEnumValueVersionCheck(const BytecodeAttrStructure &structure,
                                   llvm::raw_ostream &os);

} // namespace mlir::tblgen

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODE_ATTR_CODEGEN_H_
