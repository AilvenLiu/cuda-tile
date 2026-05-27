//===- TransformsUtils.cpp - Utils for general transforms -------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Dialect/CudaTile/Transforms/TransformsUtils.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"

using namespace mlir;
using namespace cuda_tile;

LogicalResult mlir::cuda_tile::updateSymbolName(StringAttr newName,
                                                SymbolOpInterface sym,
                                                Operation *scopeForReplacement,
                                                SymbolTable *symtab) {
  // Check the precondition - the scope for replacement must be an ancestor of
  // sym if sym has a parent op. If sym has no parent, then there's no such
  // requirement.
  if (sym->getParentOp() && !scopeForReplacement->isAncestor(sym)) {
    auto diag = emitError(sym->getLoc())
                << "attempted to update symbol names in a scope that did not "
                   "enclose the symbol itself";
    diag.attachNote(scopeForReplacement->getLoc()) << "scope defined here";
    return diag;
  }

  // If the symbol table was provided, we promised to rename all usages. Do that
  // now.
  if (symtab) {
    if (failed(symtab->rename(sym, newName)))
      return failure();
  }

  // At this point, we can update the symbol's name. If the rename happened
  // above, then this will basically do nothing.
  sym.setName(newName);

  // Grab the debuginfo location. If we don't have one, then there's nothing to
  // do besides just update the symbol's name.
  auto diLoc = dyn_cast<DILocAttr>(sym.getLoc());
  if (!diLoc)
    return success();

  // We don't have any notion of referencing a separate attribute...but there
  // may be multiple ops/locations that refer to this subprogram. We can handle
  // this by simply writing and running an AttrTypeReplacer.
  auto sp = dyn_cast_if_present<DISubprogramAttr>(diLoc.getScope());
  // If there's no subprogram on this, then we don't have to edit anything.
  if (!sp)
    return success();

  // Create the new subprogram with the new linkage name.
  auto newSP = sp.cloneWithNewLinkageName(newName);

  // Now, we have to replace *all* references to the old subprogram with the new
  // subprogram.
  AttrTypeReplacer replacer;
  replacer.addReplacement([oldSP = sp, newSP](DISubprogramAttr subprogram) {
    return subprogram == oldSP ? newSP : subprogram;
  });

  // Do the replacement within the scope.
  replacer.recursivelyReplaceElementsIn(scopeForReplacement,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true);

  return success();
}
