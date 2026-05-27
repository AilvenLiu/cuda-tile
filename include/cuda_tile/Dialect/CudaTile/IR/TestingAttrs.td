//===- TestingAttrs.td - CUDA Tile Testing Attributes ------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These attributes are used for testing bytecode versioning.
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD

include "mlir/IR/OpBase.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Testing Enum Attribute - for enum versioning tests
//===----------------------------------------------------------------------===//

// Testing enum for bytecode versioning.
def CudaTile_BytecodeTestPriority
    : CudaTileI32EnumAttr<"BytecodeTestPriority", "test priority", "250.0",
        [CudaTileI32EnumAttrCase<"Low", "Low", 0, "250.0", "low">,
         CudaTileI32EnumAttrCase<"Normal", "Normal", 1, "250.0", "normal">,
         CudaTileI32EnumAttrCase<"High", "High", 2, "250.1", "high">]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
}

def CudaTile_BytecodeTestPriorityAttr
    : CudaTileEnumAttr<CudaTile_BytecodeTestPriority, "bytecode_test_priority"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Testing Standalone Attribute - for attribute tag versioning tests
//===----------------------------------------------------------------------===//

// Simple attribute introduced in 250.1 for testing attribute tag versioning.
// This attribute just stores a single integer value.
def CudaTile_BytecodeTestValueAttr : CudaTileAttrDef<"BytecodeTestValue",
    "bytecode_test_value", "250.1"> {
  let summary = "Testing attribute for bytecode versioning";
  let description = [{
    A simple attribute introduced in version 250.1 for testing that
    attribute tag versioning works correctly.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD
