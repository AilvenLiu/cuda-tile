//===- BytecodeAttrCodeGen.cpp ----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrCodeGen.cpp - Attribute Bytecode Code Generation
//
//===----------------------------------------------------------------------===//

#include "BytecodeAttrCodeGen.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeAttrAnalysis.h"
#include "BytecodeGenUtilities.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Code Generation: AttributeTag Enum
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrTagEnum(const BytecodeAttrStructure &structure,
                                       raw_ostream &os) {
  emitSourceFileHeader("Generated AttributeTag Enum", os);
  os << "/// FROZEN at current assignments for backward compatibility.\n"
     << "/// WARNING: NEVER CHANGE THESE VALUES - they must remain stable.\n"
     << "enum class AttributeTag : uint8_t {\n";

  // bytecodeAttrs is already sorted by tag value during analysis.
  llvm::interleave(
      structure.bytecodeAttrs,
      [&](const auto &attr) {
        os << "  " << attr.attrName << " = " << attr.tagValue;
      },
      [&] { os << ",\n"; });

  os << "\n};\n";
}

//===----------------------------------------------------------------------===//
// Code Generation: Attribute Version Checking
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrVersionCheck(
    const BytecodeAttrStructure &structure, raw_ostream &os) {
  os << R"(
/// Check if an attribute tag is available in the given bytecode version.
/// Returns true if the attribute is supported, false otherwise.
inline bool isAttrTagAvailableInVersion(uint8_t tag,
                                        const BytecodeVersion &version) {
  switch (static_cast<Bytecode::AttributeTag>(tag)) {
)";

  // Generate version check for each tagged attr.
  for (const auto &attr : structure.bytecodeAttrs) {
    if (attr.skipsVersionCheck || attr.sinceVersion.empty()) {
      os << formatv(R"(  case Bytecode::AttributeTag::{0}:
    return true;
)",
                    attr.attrName);
      continue;
    }

    auto [majorStr, minorStr] = parseVersion(attr.sinceVersion);
    os << formatv(R"(  case Bytecode::AttributeTag::{0}:
    return version >= *BytecodeVersion::fromVersion({1}, {2}, 0);
)",
                  attr.attrName, majorStr, minorStr);
  }

  os << R"(  default:
    // Unknown/invalid tags are not available.
    return false;
  }
}
)";
}

//===----------------------------------------------------------------------===//
// Code Generation: Enum Type Traits
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateEnumTypeTrait(const BytecodeAttrStructure &structure,
                                         raw_ostream &os) {
  os << R"(
/// Helper type trait to check if T is one of the CUDA tile enum types.
/// Auto-generated from CudaTileI32EnumAttr/CudaTileI64EnumAttr definitions.
template <typename T>
struct is_cuda_tile_enum : std::disjunction<
)";

  llvm::interleave(
      structure.enumAttrs,
      [&](const auto &enumAttr) {
        os << "    std::is_same<T, cuda_tile::" << enumAttr.enumName << ">";
      },
      [&] { os << ",\n"; });

  os << "> {};\n";
}

void mlir::tblgen::generateEnumAttrTypeTrait(
    const BytecodeAttrStructure &structure, raw_ostream &os) {
  os << R"(
/// Helper type trait to check if T is one of the CUDA tile enum attr types.
/// Auto-generated from CudaTileI32EnumAttr/CudaTileI64EnumAttr definitions.
template <typename T>
struct is_cuda_tile_enum_attr : std::disjunction<
)";

  llvm::interleave(
      structure.enumAttrs,
      [&](const auto &enumAttr) {
        os << "    std::is_same<T, cuda_tile::" << enumAttr.attrName << ">";
      },
      [&] { os << ",\n"; });

  os << R"(> {};

/// Template declaration for symbolizeEnum.
template <typename EnumType>
static std::optional<EnumType> symbolizeEnum(uint32_t value);

/// Auto-generated symbolizeEnum specializations for CUDA tile enums.
)";

  for (const auto &enumAttr : structure.enumAttrs) {
    os << formatv(R"(template <>
[[maybe_unused]] std::optional<cuda_tile::{0}>
symbolizeEnum<cuda_tile::{0}>(uint32_t value) {{
  return cuda_tile::symbolize{0}(static_cast<int32_t>(value));
}
)",
                  enumAttr.enumName);
  }
}

//===----------------------------------------------------------------------===//
// Code Generation: Enum Attribute Version Checking
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateEnumAttrVersionCheck(
    const BytecodeAttrStructure &structure, raw_ostream &os) {
  os << R"(
/// Check if an enum attr type is available in the given bytecode version.
/// Default: always available (for types without version requirements).
template <typename AttrType>
inline bool isEnumAttrAvailableInVersion(const BytecodeVersion &) {
  return true;
}

/// Auto-generated specializations for enum attrs with version requirements.
)";

  for (const auto &enumAttr : structure.enumAttrs) {
    if (enumAttr.skipsVersionCheck || enumAttr.sinceVersion.empty())
      continue;

    auto [majorStr, minorStr] = parseVersion(enumAttr.sinceVersion);
    os << formatv(R"(template <>
inline bool isEnumAttrAvailableInVersion<cuda_tile::{0}>(
    const BytecodeVersion &version) {{
  return version >= *BytecodeVersion::fromVersion({1}, {2}, 0);
}
)",
                  enumAttr.attrName, majorStr, minorStr);
  }
}

//===----------------------------------------------------------------------===//
// Code Generation: Enum Value Version Checking (per-case)
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateEnumValueVersionCheck(
    const BytecodeAttrStructure &structure, raw_ostream &os) {
  os << R"(
/// Check if a specific enum value is available in the given bytecode version.
/// Default: always available (for enums without per-case version requirements).
template <typename EnumType>
inline bool isEnumValueAvailableInVersion(EnumType, const BytecodeVersion &) {
  return true;
}

/// Auto-generated specializations for enums with per-case version requirements.
// Only enums with version-gated cases get this specialization.
// All valid enumerants are listed explicitly. Fallthrough returns false
// to reject invalid/unknown values.

)";

  for (const auto &enumAttr : structure.enumAttrs) {
    if (enumAttr.skipsVersionCheck || enumAttr.cases.empty())
      continue;

    os << formatv(R"(template <>
inline bool isEnumValueAvailableInVersion<cuda_tile::{0}>(
    cuda_tile::{0} value, const BytecodeVersion &version) {{
  switch (value) {{
)",
                  enumAttr.enumName);

    for (const auto &enumCase : enumAttr.cases) {
      auto [majorStr, minorStr] = parseVersion(enumCase.sinceVersion);
      os << formatv(
          R"(  case cuda_tile::{0}::{1}:
    return version >= *BytecodeVersion::fromVersion({2}, {3}, 0);
)",
          enumAttr.enumName, enumCase.name, majorStr, minorStr);
    }

    os << R"(  }
  return false;
}
)";
  }
}
