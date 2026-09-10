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
// Testing Evolved Attribute - Attr that evolved from 250.0 to 250.1
//===----------------------------------------------------------------------===//

// Mirrors BytecodeTestEvolvedType for types.
def CudaTile_BytecodeTestEvolvedAttr : CudaTileAttrDef<"BytecodeTestEvolved",
    "bytecode_test_evolved_attr", "250.0"> {
  let summary = "Testing attribute for parameter evolution (250.0 -> 250.1)";
  let description = [{
    Attribute that evolved across versions:
    - Version 250.0: Only has `baseValue` (int64_t)
    - Version 250.1: Added `optionalValue` (std::optional<int64_t>)
  }];
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"int64_t", "base value">, "250.0">:$baseValue,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>",
        "optional value added in 250.1">, "250.1">:$optionalValue);
  let assemblyFormat = "`<` $baseValue (`,` $optionalValue^)? `>`";
}

#endif // CUDATILE_DIALECT_CUDATILE_IR_TESTINGATTRS_TD
