//===- TransformsUtils.h - CUDA Tile Transforms Utilities -------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_TRANSFORMS_UTILS_H
#define CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_TRANSFORMS_UTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/PointerUnion.h"

namespace mlir::cuda_tile {
/// Update the symbol name of `sym` to `newName`, as well as any attached
/// debuginfo that refers to its IR name. If `symtab` is provided, this will
/// also update the symbol's name in `symtab`, and update usages as necessary.
/// `scopeForReplacement` is a parameter that allows the caller to control which
/// IR will be walked and updated with new subprogram attributes. This scope
/// must be an ancestor (i.e. it must enclose, or be equal to) `sym`, unless
/// `sym` is unlinked in which case there is no requirement.
/// NOTE: This function has to walk all the IR in `scopeForReplacement` and
/// update *all* subprograms that may reference the old symbol, which means
/// walking every attribute and location in the scope provided. For large
/// scopes, this may be slow.
LogicalResult updateSymbolName(StringAttr newName, SymbolOpInterface sym,
                               Operation *scopeForReplacement,
                               SymbolTable *symtab = nullptr);
} // namespace mlir::cuda_tile

#endif // CUDA_TILE_DIALECT_CUDATILE_TRANSFORMS_TRANSFORMS_UTILS_H
