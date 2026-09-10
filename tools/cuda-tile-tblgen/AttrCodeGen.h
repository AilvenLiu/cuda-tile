//===- AttrCodeGen.h - Shared helpers for CudaTile tblgen -------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared utilities for CudaTile TableGen backends that inspect enum attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_CUDATILETBLGEN_ATTRCODEGEN_H_
#define CUDA_TILE_TOOLS_CUDATILETBLGEN_ATTRCODEGEN_H_

#include "llvm/TableGen/Record.h"

#include <optional>
#include <string>
#include <vector>

namespace cudatile {
namespace tblgen {

/// Return true if @p rec is (or inherits from) CudaTileI32EnumAttr.
bool isCudaTileEnumDef(const llvm::RecordKeeper &records,
                       const llvm::Record *rec);

/// Return true if @p rec wraps an I32EnumAttr with actual enumerants.
bool hasI32EnumWithCases(const llvm::Record *rec);

/// Return true if @p rec is a CudaTileEnumAttr dialect attribute wrapper.
bool isCudaTileEnumAttrWrapper(const llvm::RecordKeeper &records,
                               const llvm::Record *rec);

/// For an I32EnumAttr record, return the list of (str, symbol) pairs.
std::vector<std::pair<std::string, std::string>>
getEnumCases(const llvm::Record *enumDef);

/// Info extracted from a CudaTileEnumAttr wrapper record.
struct EnumAttrInfo {
  std::string jsonKey;   // e.g. "signedness"
  std::string attrClass; // e.g. "SignednessAttr"
  std::string enumClass; // e.g. "Signedness"
  const llvm::Record *enumDef;
};

std::optional<EnumAttrInfo>
extractEnumAttrInfo(const llvm::RecordKeeper &records,
                    const llvm::Record *wrapperRec);

} // namespace tblgen
} // namespace cudatile

#endif // CUDA_TILE_TOOLS_CUDATILETBLGEN_ATTRCODEGEN_H_
