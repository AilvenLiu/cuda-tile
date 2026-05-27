//===- Remarks.td - CUDA Tile Remark Definitions -----------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_REMARKS_TD
#define CUDATILE_DIALECT_REMARKS_TD

include "cuda_tile/Dialect/CudaTile/IR/RemarkUtils.td"

//===--------------------------------------------------------------------===//
// Diagnostic kinds
//===--------------------------------------------------------------------===//

def DK_Pass     : TileIRDiagnosticKind<"Pass",     0, "Succeeded">;
def DK_Miss     : TileIRDiagnosticKind<"Miss",     1, "Missed">;
def DK_Fail     : TileIRDiagnosticKind<"Fail",     2, "Failed">;
def DK_Analysis : TileIRDiagnosticKind<"Analysis", 3, "Analysis">;

//===--------------------------------------------------------------------===//
// Remark Categories
//===--------------------------------------------------------------------===//

def CategoryTensorCore : TileIRCategory<"TensorCore", 1, "Tensor-core">;
def CategoryMemory     : TileIRCategory<"Memory",     2, "Memory">;
//===--------------------------------------------------------------------===//
// Remarks
//===--------------------------------------------------------------------===//

// Tensor Core Remarks
def RemarkTensorCoreMMA
  : TileIRRemark<"RemarkTensorCoreMMA", DK_Pass, CategoryTensorCore, 1, "MMA operation successfully optimized to use Tensor Cores">;

def RemarkTensorCoreMMAFailed
  : TileIRRemark<"RemarkTensorCoreMMA", DK_Fail, CategoryTensorCore, 2, "MMA operation failed to optimize to use Tensor Cores, it is using FMA instructions instead">;

// Memory Remarks
def RemarkMemoryLoadOptimized
  : TileIRRemark<"RemarkMemoryLoadOptimized", DK_Pass, CategoryMemory, 1, "Load operation successfully optimized to use TMA">;

def RemarkMemoryStoreOptimized
  : TileIRRemark<"RemarkMemoryStoreOptimized", DK_Pass, CategoryMemory, 2, "Store operation successfully optimized to use TMA">;

//===--------------------------------------------------------------------===//
// Remark Attribute Definitions
//===--------------------------------------------------------------------===//

def RemarkID
    : I32EnumAttr<"RemarkID", "All TileIR remark IDs",
                  [RemarkTensorCoreMMA, 
                   RemarkMemoryLoadOptimized, 
                   RemarkMemoryStoreOptimized,
                   RemarkTensorCoreMMAFailed
                   /*, ...*/]> {
  let cppNamespace = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}
//===--------------------------------------------------------------------===//
// Attribute Definitions
//===--------------------------------------------------------------------===//

def DiagnosticKind : I32EnumAttr<"DiagnosticKind",
      "Kinds of optimization diagnostics",
      [DK_Pass, DK_Miss, DK_Fail, DK_Analysis]> {
  let cppNamespace       = "::mlir::cuda_tile::remark";
  let genSpecializedAttr = 0;
}

def CategoryID : I32EnumAttr<"CategoryID",
                             "TileIR remark categories",
                             [CategoryTensorCore, 
                              CategoryMemory
                             ]> {
  let cppNamespace        = "::mlir::cuda_tile::remark";
  let genSpecializedAttr  = 0;
}

#endif
