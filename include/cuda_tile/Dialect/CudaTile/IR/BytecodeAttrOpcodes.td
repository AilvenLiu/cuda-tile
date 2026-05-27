//===- BytecodeAttrOpcodes.td - Bytecode Attribute Tags ----*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bytecode Attribute Tag Assignments for CudaTile Attributes
// This file defines explicit attribute tag assignments to ensure backward
// compatibility across versions.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD

include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"

//===----------------------------------------------------------------------===//
// Attribute Tag Assignment Class
//===----------------------------------------------------------------------===//

/// Bytecode attribute tag assignment.
/// Version information is looked up from CudaTileAttrAlias (for MLIR builtins)
/// or CudaTileAttrDef (for CudaTile-specific attrs) in AttrDefs.td.
class BytecodeAttrTag<string attrName, int tag> {
  string cppAttrName = attrName;
  int attrTagValue = tag;
}

//===----------------------------------------------------------------------===//
// Attribute Tags - FROZEN for backward compatibility.
// Listed in tag value order.
//===----------------------------------------------------------------------===//

def : BytecodeAttrTag<"Integer", 1>;
def : BytecodeAttrTag<"Float", 2>;
def : BytecodeAttrTag<"Bool", 3>;
def : BytecodeAttrTag<"Type", 4>;
def : BytecodeAttrTag<"String", 5>;
def : BytecodeAttrTag<"Array", 6>;
def : BytecodeAttrTag<"DenseElements", 7>;
def : BytecodeAttrTag<"DivBy", 8>;
def : BytecodeAttrTag<"SameElements", 9>;
def : BytecodeAttrTag<"Dictionary", 10>;
def : BytecodeAttrTag<"OptimizationHints", 11>;
def : BytecodeAttrTag<"Bounded", 12>;

#ifdef TILE_IR_INCLUDE_TESTS
// Testing Attribute Tags - Only available in testing builds.
def : BytecodeAttrTag<"BytecodeTestValue", 200>;
#endif // TILE_IR_INCLUDE_TESTS

#endif // CUDATILE_DIALECT_CUDATILE_IR_BYTECODE_ATTR_OPCODES_TD
