//===- BytecodeGenUtilities.h - Bytecode Gen Utilities ----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities used across multiple bytecode generation
// TableGen backends for cuda_tile operations.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_
#define CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_

#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/TableGen/Error.h"

#include <optional>
#include <string>

namespace mlir {
namespace tblgen {

/// Parse version string into major/minor components.
/// Returns StringRefs pointing into the input string (or "0" for missing
/// minor).
std::pair<StringRef, StringRef> parseVersion(StringRef version);

/// Check if a parsed version represents the minimum supported version (13.1).
inline bool isMinimumVersion(StringRef majorStr, StringRef minorStr) {
  return majorStr == "13" && minorStr == "1";
}

/// Check if version A is greater than version B.
/// Returns false if either version string is empty or invalid.
/// Version components are numeric strings like "13", "2" (not "13.2").
inline bool isVersionGreater(StringRef majorA, StringRef minorA,
                             StringRef majorB, StringRef minorB) {
  unsigned majA, minA, majB, minB;
  // getAsInteger returns true on failure (non-numeric or overflow).
  if (majorA.getAsInteger(10, majA) || minorA.getAsInteger(10, minA) ||
      majorB.getAsInteger(10, majB) || minorB.getAsInteger(10, minB)) {
    return false;
  }
  return majA > majB || (majA == majB && minA > minB);
}

/// Extract version information from an attribute's TableGen metadata.
std::pair<StringRef, StringRef>
extractVersionFromAttribute(const NamedAttribute &namedAttr,
                            const Operator &op);

/// Extract the default value from an attribute if it has one.
std::optional<std::string> extractDefaultValue(const NamedAttribute &namedAttr);

/// If the given attribute is decorated with `RequireSameOperandRank`,
/// return the name of the operand whose rank should be used to size the
/// fixup value. Returns std::nullopt otherwise.
std::optional<std::string>
extractSameOperandRankName(const NamedAttribute &namedAttr, const Operator &op);

/// Extract the version string from an operation's metadata.
std::string extractVersionFromOperation(const Operator &op);

/// Get version-ordered bit assignments for optional fields.
/// Returns map from field name to bit position, and optionally the earliest
/// version among all optional fields (if any exist).
std::pair<llvm::StringMap<size_t>,
          std::optional<std::pair<std::string, std::string>>>
getVersionOrderedBitAssignments(const Operator &op);

/// Extract version information from an operand's TableGen metadata.
std::pair<StringRef, StringRef> extractVersionFromOperand(unsigned operandIndex,
                                                          const Operator &op);

/// Extract version information from a result's TableGen metadata.
std::pair<StringRef, StringRef> extractVersionFromResult(unsigned resultIndex,
                                                         const Operator &op);

/// Shared structure to capture version info for result
/// serialization/deserialization.
struct ResultVersionInfo {
  std::string majorStr, minorStr;
  std::string name;
  bool requiresVersionCheck;

  ResultVersionInfo(int idx, const NamedTypeConstraint &result,
                    const Operator &op, const std::string &opVersion)
      : name(result.name.str()) {
    std::tie(majorStr, minorStr) = extractVersionFromResult(idx, op);
    requiresVersionCheck = (majorStr + "." + minorStr != opVersion);

    // Validate that required results added after operation version are
    // buildable.
    if (requiresVersionCheck) {
      std::optional<StringRef> builderCall = result.constraint.getBuilderCall();
      if (!builderCall.has_value())
        llvm::PrintFatalError("Required result '" + result.name.str() +
                              "' in operation '" + op.getOperationName() +
                              "' was introduced after the operation (version " +
                              majorStr + "." + minorStr + " vs " + opVersion +
                              ") and has non-buildable type constraint '" +
                              result.constraint.getDefName().str() +
                              "'. Results added after operation version must "
                              "have buildable types.");
    }
  }
};

} // namespace tblgen
} // namespace mlir

#endif // CUDA_TILE_TOOLS_TBLGEN_BYTECODEGEN_UTILITIES_H_
