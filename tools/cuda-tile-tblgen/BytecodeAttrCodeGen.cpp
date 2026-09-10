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

#include "mlir/Support/IndentedOstream.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
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

//===----------------------------------------------------------------------===//
// Code Generation: Attribute Version Map (for MinVersionAnalyzer)
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrVersionMap(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(namespace mlir::cuda_tile::detail {

/// Returns the minimum bytecode version required for an Attribute.
/// Returns nullopt for unsupported or unversioned attributes, or those
/// at the minimum supported version.
inline std::optional<BytecodeVersion>
getAttrMinVersion(Attribute attr) {
  return llvm::TypeSwitch<Attribute, std::optional<BytecodeVersion>>(attr)
)";

  for (const auto &attr : structure.bytecodeAttrs) {
    if (attr.skipsVersionCheck || attr.sinceVersion.empty()) {
      continue;
    }
    auto [majorStr, minorStr] = parseVersion(attr.sinceVersion);
    if (isMinimumVersion(majorStr, minorStr)) {
      continue;
    }
    os << formatv(R"(      .Case<cuda_tile::{0}Attr>([](auto) {{
        return BytecodeVersion::fromVersion({1}, {2}, 0);
      })
)",
                  attr.attrName, majorStr, minorStr);
  }

  os << R"(      .Default([](Attribute) -> std::optional<BytecodeVersion> {
        return std::nullopt;
      });
}

} // namespace mlir::cuda_tile::detail
)";
}

//===----------------------------------------------------------------------===//
// Code Generation: Enum Value Version Map (for MinVersionAnalyzer)
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateEnumValueVersionMap(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(namespace mlir::cuda_tile::detail {

/// Default: returns nullopt for enums without per-case version requirements.
template <typename EnumType>
inline std::optional<BytecodeVersion> getEnumValueMinVersion(EnumType) {
  return std::nullopt;
}

)";

  // Accumulate TypeSwitch cases while generating specializations (single pass).
  std::string typeSwitchCasesStr;
  llvm::raw_string_ostream typeSwitchCases(typeSwitchCasesStr);

  // Generate specializations for enums with per-case version requirements.
  for (const auto &enumAttr : structure.enumAttrs) {
    if (enumAttr.skipsVersionCheck || enumAttr.cases.empty()) {
      continue;
    }

    // Build switch cases for non-minimum version enum values only.
    std::string switchCasesStr;
    llvm::raw_string_ostream switchCases(switchCasesStr);
    for (const auto &enumCase : enumAttr.cases) {
      auto [majorStr, minorStr] = parseVersion(enumCase.sinceVersion);
      if (isMinimumVersion(majorStr, minorStr)) {
        continue;
      }
      switchCases << formatv(R"(  case cuda_tile::{0}::{1}:
    return BytecodeVersion::fromVersion({2}, {3}, 0);
)",
                             enumAttr.enumName, enumCase.name, majorStr,
                             minorStr);
    }

    if (switchCasesStr.empty()) {
      continue;
    }

    // Output specialization and accumulate TypeSwitch case.
    os << formatv(R"(template <>
inline std::optional<BytecodeVersion>
getEnumValueMinVersion<cuda_tile::{0}>(cuda_tile::{0} value) {{
  switch (value) {{
)",
                  enumAttr.enumName)
       << switchCasesStr << R"(  default:
    return std::nullopt;
  }
}

)";

    typeSwitchCases << formatv(
        R"(      .Case<cuda_tile::{0}>([](auto enumAttr) {{
        return getEnumValueMinVersion(enumAttr.getValue());
      })
)",
        enumAttr.attrName);
  }

  // Generate the unified dispatch function with accumulated cases.
  os << R"(/// Returns the minimum bytecode version for an enum attribute's value.
inline std::optional<BytecodeVersion>
getEnumAttrValueMinVersion(Attribute attr) {
  return llvm::TypeSwitch<Attribute, std::optional<BytecodeVersion>>(attr)
)" << typeSwitchCasesStr
     << R"(      .Default([](Attribute) -> std::optional<BytecodeVersion> {
        return std::nullopt;
      });
}

} // namespace mlir::cuda_tile::detail
)";
}

//===----------------------------------------------------------------------===//
// Code Generation: Attribute Parameter Version Map (for MinVersionAnalyzer)
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrParamVersionMap(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(namespace mlir::cuda_tile::detail {

/// Returns the minimum bytecode version required based on attribute parameters.
/// Checks if optional parameters added in later versions are present.
/// Returns nullopt if no versioned parameters are present.
inline std::optional<BytecodeVersion>
getAttrParamMinVersion(Attribute attr) {
  return llvm::TypeSwitch<Attribute, std::optional<BytecodeVersion>>(attr)
)";

  // Generate cases for attrs with versioned optional parameters.
  for (const auto &attr : structure.serializableAttrs) {
    if (attr.skipVersionCheck) {
      continue;
    }

    // Find optional parameters with version > attribute's base version.
    auto [attrMajor, attrMinor] = parseVersion(attr.sinceVersion);
    bool hasVersionedParams = false;
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        continue;
      }
      auto [paramMajor, paramMinor] = parseVersion(param.sinceVersion);
      if (isVersionGreater(paramMajor, paramMinor, attrMajor, attrMinor)) {
        hasVersionedParams = true;
        break;
      }
    }

    if (!hasVersionedParams) {
      continue;
    }

    // Generate the Case with inline parameter checks.
    // Note: attrName already includes "Attr" suffix (e.g.,
    // "BytecodeTestEvolvedAttr").
    os << formatv("      .Case<cuda_tile::{0}>([](cuda_tile::{0} a)"
                  " -> std::optional<BytecodeVersion> {{\n",
                  attr.attrName);

    // Generate checks for each versioned optional parameter.
    bool first = true;
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        continue;
      }
      auto [paramMajor, paramMinor] = parseVersion(param.sinceVersion);
      if (!isVersionGreater(paramMajor, paramMinor, attrMajor, attrMinor)) {
        continue;
      }

      if (first) {
        os << formatv("        if (a.{0}())\n", param.accessorName);
        os << formatv(
            "          return BytecodeVersion::fromVersion({0}, {1}, 0);\n",
            paramMajor, paramMinor);
        first = false;
      } else {
        os << formatv("        else if (a.{0}())\n", param.accessorName);
        os << formatv(
            "          return BytecodeVersion::fromVersion({0}, {1}, 0);\n",
            paramMajor, paramMinor);
      }
    }

    os << "        return std::nullopt;\n";
    os << "      })\n";
  }

  os << R"(      .Default([](Attribute) -> std::optional<BytecodeVersion> {
        return std::nullopt;
      });
}

/// Returns the highest bytecode version contributed by an attribute value: the
/// attribute's own version, its enum case (if any), and its evolved parameters.
/// Returns nullopt for a null attribute or one that stays at the minimum
/// supported version. Shared by the per-operation attribute scan and by
/// getTypeMinVersionRecursive, so an attribute is scored identically whether it
/// appears on an operation or nested inside a type parameter.
inline std::optional<BytecodeVersion>
getAttrMinVersionRecursive(Attribute attr) {
  std::optional<BytecodeVersion> maxVersion;
  auto updateMax = [&](std::optional<BytecodeVersion> version) {
    if (version && (!maxVersion || *version > *maxVersion))
      maxVersion = version;
  };
  updateMax(getAttrMinVersion(attr));
  updateMax(getEnumAttrValueMinVersion(attr));
  updateMax(getAttrParamMinVersion(attr));
  return maxVersion;
}

} // namespace mlir::cuda_tile::detail
)";
}

// Attribute Serialization/Deserialization Templates
//===----------------------------------------------------------------------===//
//
// These templates mirror BytecodeTypeCodeGen patterns for consistency.
// Wire format:
//   [tag:varint] [flags:varint (if optional params)] [params...]
//
// For attrs with requiredParamsFirst=true (legacy: DivByAttr only):
//   [tag:varint] [required params...] [flags:varint] [optional params...]
//
// Templates are organized by Kind (wire format), with signedness handled
// via isSigned flag for VarInt types.
//
//===----------------------------------------------------------------------===//

// VarInt serialization templates
// {0}: getter call (e.g., "attr.getValue()")
static const char *const serializeVarIntTemplate = "writer.writeVarInt({0});\n";
static const char *const serializeSignedVarIntTemplate =
    "writer.writeSignedVarInt({0});\n";
static const char *const serializeAPIntTemplate =
    "writer.writeSignedVarInt({0}.getSExtValue());\n";

// DenseArray serialization template
static const char *const serializeDenseArrayTemplate =
    "writer.writeLEVarSize({0}.asArrayRef());\n";

// NestedAttr serialization template - inline encoding (no tag prefix).
// Matches the original wire format where the attribute type is implied by
// the parameter declaration. Use AttributeArray (or call
// writeSelfContainedAttribute directly) when self-contained encoding is needed.
static const char *const serializeNestedAttrTemplate = R"(
if (failed(writeSingleAttribute(op, "nested", {0}, writer,
                                typeMgr, constMgr, strMgr,
                                /*isSelfContained=*/false)))
  return failure();
)";

// PolymorphicNestedAttr serialization template - self-contained encoding
// (tag + payload) for parameters whose declared C++ type is an
// AttrInterface, where the concrete attribute is not statically known.
// {0}: getter call.
static const char *const serializePolymorphicNestedAttrTemplate = R"(
if (failed(writeSelfContainedAttribute(op, "polymorphic_nested", {0}, writer,
                                       typeMgr, constMgr, strMgr)))
  return failure();
)";

static const char *const serializeTypeAttrTemplate = R"(
if (failed(typeMgr.writeTypeIndex({0}.getValue(), writer)))
  return failure();
)";

// Enum values are serialized as uint64_t so both I32EnumAttr and I64EnumAttr
// round-trip losslessly.
static const char *const serializeEnumTemplate = R"(
writer.writeVarInt(static_cast<uint64_t>({0}));
)";

static const char *const serializeAttrArrayTemplate = R"(
writer.writeVarInt({0}.size());
for (Attribute elementAttr : {0}) {{
  if (failed(writeSelfContainedAttribute(op, "arrayElement", elementAttr,
                                         writer, typeMgr, constMgr, strMgr)))
    return failure();
}
)";

static const char *const serializeStringAttrTemplate =
    "writer.writeVarInt(strMgr.getStringIndex({0}.getValue()));\n";

// StringRef serialization template
static const char *const serializeStringRefTemplate =
    "writer.writeVarInt(strMgr.getStringIndex({0}));\n";

// VarInt deserialization templates
// {0}: variable name
static const char *const readVarIntTemplate = R"(
if (failed(reader.readVarInt({0})))
  return reader.emitError() << "failed to read {0}";
)";
// EncodingReader::readSignedVarInt takes uint64_t& (zigzag decode).
static const char *const readSignedVarIntTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readSignedVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
{0} = static_cast<int64_t>({0}_raw);
)";
static const char *const readAPIntTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readSignedVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
{0} = ::llvm::APInt(64, static_cast<int64_t>({0}_raw), /*isSigned=*/true);
)";

// DenseArray deserialization template
// {0}: variable name, {1}: element type (e.g. int64_t, float), {2}: attr type
static const char *const readDenseArrayTemplate = R"(
SmallVector<{1}, 4> {0}_data;
if (failed(reader.readLEVarSize({0}_data)))
  return reader.emitError() << "failed to read {0} data";
{0} = {2}::get(context, {0}_data);
)";

// NestedAttr deserialization template - inline encoding (no tag).
// {0}: variable name.
static const char *const readNestedAttrTemplate = R"(
if (failed(readNestedAttribute(reader, {0})))
  return failure();
)";

// PolymorphicNestedAttr deserialization template: self-contained
// encoding (tag + payload). Reads a generic Attribute via tag dispatch
// then narrows to the interface type.
// {0}: variable name. {1}: declared C++ interface type.
static const char *const readPolymorphicNestedAttrTemplate = R"(
{
  Attribute {0}_self_contained;
  if (failed(readNestedAttribute(reader, {0}_self_contained)))
    return failure();
  {0} = ::llvm::dyn_cast<{1}>({0}_self_contained);
  if (!{0})
    return reader.emitError() << "expected attribute implementing '{1}' "
                              << "for parameter '{0}', got "
                              << {0}_self_contained;
}
)";

static const char *const readTypeAttrTemplate = R"(
Type {0}_type = types.readAndGetType(reader);
if (!{0}_type)
  return reader.emitError() << "failed to read {0} type";
{0} = TypeAttr::get({0}_type);
)";

// Reads an enum value as uint64_t then narrows it to the enum's underlying
// type. Supports both I32EnumAttr and I64EnumAttr.
static const char *const readEnumTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
using {0}_underlying = std::underlying_type_t<{1}>;
if ({0}_raw > static_cast<uint64_t>(std::numeric_limits<{0}_underlying>::max()))
  return reader.emitError() << "enum value for {0} (" << {0}_raw
                            << ") does not fit in "
                            << (sizeof({0}_underlying) * 8) << "-bit storage";
auto {0}_sym = symbolizeEnum<{1}>(static_cast<{0}_underlying>({0}_raw));
if (!{0}_sym)
  return reader.emitError() << "invalid enum value for {0}: " << {0}_raw;
{0} = *{0}_sym;
)";

static const char *const readAttrArrayTemplate = R"(
uint64_t {0}_size;
if (failed(reader.readVarInt({0}_size,
                             std::numeric_limits<uint32_t>::max() - 1)))
  return reader.emitError() << "failed to read {0} size";
{0}.reserve({0}_size);
for (uint64_t i = 0; i < {0}_size; ++i) {{
  Attribute {0}_element;
  if (failed(readNestedAttribute(reader, {0}_element)))
    return reader.emitError() << "failed to read {0} element " << i;
  {0}.push_back({0}_element);
}
)";

// Finalizes an AttributeArray deserialization into an ArrayAttr.
// {0}: variable name, {1}: storage name (SmallVector<Attribute>).
static const char *const finalizeAttrArrayAsArrayAttrTemplate =
    "{0} = ArrayAttr::get(context, {1});\n";

// Finalizes an AttributeArray deserialization into an
// ArrayRef<ConcreteAttr> by casting each element.
// {0}: variable name, {1}: storage name, {2}: concrete element type.
static const char *const finalizeAttrArrayAsArrayRefCastTemplate = R"(
SmallVector<{2}, 4> {0}_typed_storage;
{0}_typed_storage.reserve({1}.size());
for (Attribute a : {1})
  {0}_typed_storage.push_back(cast<{2}>(a));
{0} = {0}_typed_storage;
)";

// Finalizes an AttributeArray deserialization into an ArrayRef<Attribute>
// (no cast required).
// {0}: variable name, {1}: storage name.
static const char *const finalizeAttrArrayAsArrayRefTemplate = "{0} = {1};\n";

static const char *const readStringAttrTemplate = R"(
uint64_t {0}_idx;
if (failed(reader.readVarInt({0}_idx)))
  return reader.emitError() << "failed to read {0} string index";
{0} = getStringAttr(reader, {0}_idx);
if (!{0})
  return reader.emitError() << "failed to resolve {0} string index";
)";

// StringRef deserialization template
static const char *const readStringRefTemplate = R"(
uint64_t {0}_idx;
if (failed(reader.readVarInt({0}_idx)))
  return reader.emitError() << "failed to read {0} string index";
StringAttr {0}_attr = getStringAttr(reader, {0}_idx);
if (!{0}_attr)
  return reader.emitError() << "failed to resolve {0} string index";
{0} = {0}_attr.getValue();
)";

//===----------------------------------------------------------------------===//
// Helper Functions for Attr Serialization
//===----------------------------------------------------------------------===//

/// Generate version check for attr serialization.
static std::string
generateAttrVersionCheck(unsigned indent, StringRef version, StringRef attrName,
                         StringRef versionVar = "config.bytecodeVersion",
                         StringRef contextExpr = "attr.getContext()") {
  auto [majorStr, minorStr] = parseVersion(version);
  std::string indentStr(indent, ' ');
  return formatv("{0}if ({1} < *BytecodeVersion::fromVersion({2}, {3}, 0))\n"
                 "{0}  return ::emitError(UnknownLoc::get({4}))\n"
                 "{0}         << \"attribute '{5}' requires bytecode version "
                 "{6}+, targeting \" << {1}.toString();\n",
                 indentStr, versionVar, majorStr, minorStr, contextExpr,
                 attrName, version)
      .str();
}

//===----------------------------------------------------------------------===//
// Parameter Serialization (Attr)
//===----------------------------------------------------------------------===//

static void
generateAttrParameterSerialization(const BytecodeAttrParameter &param,
                                   llvm::raw_ostream &os, StringRef indent) {
  std::string getterCall = "attr." + param.accessorName + "()";
  // For optional params, caller wraps in presence check; dereference the value.
  if (param.isOptional && param.kind == BytecodeAttrParameter::Kind::VarInt) {
    getterCall = "*" + getterCall;
  }

  mlir::raw_indented_ostream ios(os);

  switch (param.kind) {
  case BytecodeAttrParameter::Kind::VarInt:
    if (param.isSigned) {
      ios.printReindented(
          formatv(serializeSignedVarIntTemplate, getterCall).str(), indent);
    } else {
      ios.printReindented(formatv(serializeVarIntTemplate, getterCall).str(),
                          indent);
    }
    break;
  case BytecodeAttrParameter::Kind::APInt:
    ios.printReindented(formatv(serializeAPIntTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::DenseArray:
    ios.printReindented(formatv(serializeDenseArrayTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::EnumValue:
    ios.printReindented(formatv(serializeEnumTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::TypeAttr:
    ios.printReindented(formatv(serializeTypeAttrTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::NestedAttr:
    ios.printReindented(formatv(serializeNestedAttrTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::PolymorphicNestedAttr:
    ios.printReindented(
        formatv(serializePolymorphicNestedAttrTemplate, getterCall).str(),
        indent);
    break;
  case BytecodeAttrParameter::Kind::AttributeArray:
    ios.printReindented(formatv(serializeAttrArrayTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::StringAttr:
    ios.printReindented(formatv(serializeStringAttrTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeAttrParameter::Kind::StringRef:
    ios.printReindented(formatv(serializeStringRefTemplate, getterCall).str(),
                        indent);
    break;
  }
}

//===----------------------------------------------------------------------===//
// Parameter Deserialization (Attr)
//===----------------------------------------------------------------------===//

/// Helper to get array element type and MLIR attr type from C++ type string.
static std::pair<StringRef, StringRef> getDenseArrayInfo(StringRef cppType) {
  if (cppType.contains("DenseBoolArrayAttr")) {
    return {"bool", "DenseBoolArrayAttr"};
  }
  if (cppType.contains("DenseI8ArrayAttr")) {
    return {"int8_t", "DenseI8ArrayAttr"};
  }
  if (cppType.contains("DenseI16ArrayAttr")) {
    return {"int16_t", "DenseI16ArrayAttr"};
  }
  if (cppType.contains("DenseI32ArrayAttr")) {
    return {"int32_t", "DenseI32ArrayAttr"};
  }
  if (cppType.contains("DenseI64ArrayAttr")) {
    return {"int64_t", "DenseI64ArrayAttr"};
  }
  if (cppType.contains("DenseF32ArrayAttr")) {
    return {"float", "DenseF32ArrayAttr"};
  }
  if (cppType.contains("DenseF64ArrayAttr")) {
    return {"double", "DenseF64ArrayAttr"};
  }
  PrintFatalError("Unknown DenseArrayAttr type: " + cppType.str());
}

static void
generateAttrParameterDeserialization(const BytecodeAttrParameter &param,
                                     llvm::raw_ostream &os, StringRef indent,
                                     bool declareVariable) {
  mlir::raw_indented_ostream ios(os);
  std::string varName = param.name;

  switch (param.kind) {
  case BytecodeAttrParameter::Kind::VarInt:
    if (param.isOptional) {
      // For optional VarInt, read into temp then assign to std::optional.
      if (declareVariable) {
        os << indent << formatv("{0} {1};\n", param.cppType, varName);
      }
      std::string baseType = param.isSigned ? "int64_t" : "uint64_t";
      os << indent << "{\n";
      os << indent << "  " << baseType << " " << varName << "_val;\n";
      if (param.isSigned) {
        ios.printReindented(
            formatv(readSignedVarIntTemplate, varName + "_val").str(),
            (indent + "  ").str());
      } else {
        ios.printReindented(formatv(readVarIntTemplate, varName + "_val").str(),
                            (indent + "  ").str());
      }
      os << indent << "  " << varName << " = " << varName << "_val;\n";
      os << indent << "}\n";
    } else {
      // Regular VarInt
      if (declareVariable) {
        os << indent << formatv("{0} {1};\n", param.cppType, varName);
      }
      if (param.isSigned) {
        ios.printReindented(formatv(readSignedVarIntTemplate, varName).str(),
                            indent);
      } else {
        ios.printReindented(formatv(readVarIntTemplate, varName).str(), indent);
      }
    }
    break;

  case BytecodeAttrParameter::Kind::APInt:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(formatv(readAPIntTemplate, varName).str(), indent);
    break;

  case BytecodeAttrParameter::Kind::DenseArray: {
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    auto [elemType, attrType] = getDenseArrayInfo(param.cppType);
    ios.printReindented(
        formatv(readDenseArrayTemplate, varName, elemType, attrType).str(),
        indent);
    break;
  }

  case BytecodeAttrParameter::Kind::EnumValue:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(formatv(readEnumTemplate, varName, param.cppType).str(),
                        indent);
    break;

  case BytecodeAttrParameter::Kind::TypeAttr:
    if (declareVariable) {
      os << indent << formatv("TypeAttr {0};\n", varName);
    }
    ios.printReindented(formatv(readTypeAttrTemplate, varName).str(), indent);
    break;

  case BytecodeAttrParameter::Kind::NestedAttr:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(formatv(readNestedAttrTemplate, varName).str(), indent);
    break;

  case BytecodeAttrParameter::Kind::PolymorphicNestedAttr:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(
        formatv(readPolymorphicNestedAttrTemplate, varName, param.cppType)
            .str(),
        indent);
    break;

  case BytecodeAttrParameter::Kind::AttributeArray: {
    bool isArrayAttr = StringRef(param.cppType).contains("ArrayAttr");
    bool isArrayRef = StringRef(param.cppType).contains("ArrayRef");

    // Extract element type from ArrayRef<ElementType>, if present.
    std::string elemType = "Attribute";
    if (isArrayRef) {
      StringRef typeStr(param.cppType);
      size_t begin = typeStr.find('<');
      size_t end = typeStr.rfind('>');
      if (begin != StringRef::npos && end != StringRef::npos && begin < end) {
        elemType = typeStr.slice(begin + 1, end).trim().str();
      }
    }
    bool needsCast = isArrayRef && elemType != "Attribute" &&
                     elemType != "::mlir::Attribute";

    // Declaration (only when declareVariable is true — optional parameter
    // sites declare the storage separately).
    if (declareVariable) {
      if (isArrayAttr) {
        os << indent << formatv("ArrayAttr {0};\n", varName);
      } else if (isArrayRef) {
        os << indent << param.cppType << " " << varName << ";\n";
      }
    }

    // Deserialization (must always run regardless of declareVariable).
    std::string storageName =
        isArrayAttr ? varName + "_storage"
                    : (isArrayRef ? varName + "_attr_storage" : varName);
    os << indent << "SmallVector<Attribute, 4> " << storageName << ";\n";
    ios.printReindented(formatv(readAttrArrayTemplate, storageName).str(),
                        indent);

    if (isArrayAttr) {
      ios.printReindented(
          formatv(finalizeAttrArrayAsArrayAttrTemplate, varName, storageName)
              .str(),
          indent);
    } else if (isArrayRef) {
      if (needsCast) {
        ios.printReindented(formatv(finalizeAttrArrayAsArrayRefCastTemplate,
                                    varName, storageName, elemType)
                                .str(),
                            indent);
      } else {
        ios.printReindented(
            formatv(finalizeAttrArrayAsArrayRefTemplate, varName, storageName)
                .str(),
            indent);
      }
    }
    break;
  }

  case BytecodeAttrParameter::Kind::StringAttr:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(formatv(readStringAttrTemplate, varName).str(), indent);
    break;

  case BytecodeAttrParameter::Kind::StringRef:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, varName);
    }
    ios.printReindented(formatv(readStringRefTemplate, varName).str(), indent);
    break;
  }
}

//===----------------------------------------------------------------------===//
// Attr Serializer Generation
//===----------------------------------------------------------------------===//

/// Check if a parameter is evolved (added after the attr was introduced).
static bool isEvolvedParam(const CudaTileSerializableAttr &attr,
                           const BytecodeAttrParameter &param) {
  return !attr.skipVersionCheck && !param.sinceVersion.empty() &&
         param.sinceVersion != attr.sinceVersion;
}

/// Generate serialization function for a single attribute.
static void generateCudaTileAttrSerializer(const CudaTileSerializableAttr &attr,
                                           llvm::raw_ostream &os) {
  os << formatv(R"(
// Writer for Attr: {0}
// Note: Does NOT write the tag - caller handles tag based on context.
LogicalResult serialize{0}Data(Operation *op, {1} attr,
                               EncodingWriter &writer) {{
)",
                attr.attrName, attr.qualifiedCppName);

  // Version check for the attr itself.
  if (!attr.skipVersionCheck && !attr.sinceVersion.empty()) {
    os << generateAttrVersionCheck(2, attr.sinceVersion, attr.attrName);
  }

  // For requiredParamsFirst attrs (legacy: DivByAttr only), write required
  // params first.
  if (attr.requiredParamsFirst) {
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        generateAttrParameterSerialization(param, os, "  ");
      }
    }
  }

  // Validate evolved optional params: error if value is present but targeting
  // an older version.
  for (const auto &param : attr.parameters) {
    if (param.isOptional && isEvolvedParam(attr, param)) {
      auto [majorStr, minorStr] = parseVersion(param.sinceVersion);
      os << formatv(R"(
  if (config.bytecodeVersion < *BytecodeVersion::fromVersion({0}, {1}, 0) &&
      attr.{2}().has_value())
    return ::emitError(UnknownLoc::get(attr.getContext()))
           << "attribute parameter '{3}' requires bytecode version {4}+, "
              "but targeting " << config.bytecodeVersion.toString();
)",
                    majorStr, minorStr, param.accessorName, param.name,
                    param.sinceVersion);
    }
  }

  // Generate optional flags if attribute has optional params.
  if (attr.hasOptionalParams) {
    os << R"(
  // Build and write optional parameter flags (varint-encoded bitmask).
  uint64_t optionalFlags = 0;
)";
    unsigned bitIndex = 0;
    for (const auto &param : attr.parameters) {
      if (param.isOptional) {
        os << formatv(
            R"(  if (attr.{0}().has_value()) optionalFlags |= (1ULL << {1});
)",
            param.accessorName, bitIndex++);
      }
    }
    os << "  writer.writeVarInt(optionalFlags);\n";
  }

  // For non-requiredParamsFirst attrs, write required params after flags.
  if (!attr.requiredParamsFirst) {
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        generateAttrParameterSerialization(param, os, "  ");
      }
    }
  }

  // Serialize optional parameters (evolved params already validated above).
  for (const auto &param : attr.parameters) {
    if (param.isOptional) {
      os << formatv(R"(  if (attr.{0}().has_value()) {{
)",
                    param.accessorName);
      generateAttrParameterSerialization(param, os, "    ");
      os << "  }\n";
    }
  }

  os << "  return success();\n}\n";
}

void mlir::tblgen::generateAttrSerializers(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  emitSourceFileHeader("Generated Attribute Serialization Functions", os);
  os << R"(
// Auto-generated attribute serializers.
// Format: [tag:varint] [flags:varint if optional] [params...]
// For requiredParamsFirst attrs: [tag] [required params] [flags] [optional params]

)";
  for (const auto &attr : structure.serializableAttrs) {
    generateCudaTileAttrSerializer(attr, os);
  }
}

//===----------------------------------------------------------------------===//
// Attr Deserializer Generation
//===----------------------------------------------------------------------===//

/// Generate deserialization function for a single attribute.
/// These are member functions that access: context, fileVersion, and
/// readNestedAttribute() for nested attrs.
static void
generateCudaTileAttrDeserializer(const CudaTileSerializableAttr &attr,
                                 llvm::raw_ostream &os) {
  os << formatv(R"(
// Reader for Attr: {0}
LogicalResult parse{0}Data(EncodingReader &reader, Attribute &result) {{
)",
                attr.attrName);

  // Version check.
  if (!attr.skipVersionCheck && !attr.sinceVersion.empty()) {
    os << generateAttrVersionCheck(2, attr.sinceVersion, attr.attrName,
                                   "fileVersion", "context");
  }

  // For requiredParamsFirst attrs (legacy: DivByAttr only), read required
  // params first.
  if (attr.requiredParamsFirst) {
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        generateAttrParameterDeserialization(param, os, "  ",
                                             /*declareVariable=*/true);
      }
    }
  }

  // Read optional flags if attribute has optional params.
  if (attr.hasOptionalParams) {
    os << R"(
  // Read optional parameter flags (varint-encoded bitmask).
  uint64_t optionalFlags = 0;
  if (failed(reader.readVarInt(optionalFlags)))
    return reader.emitError() << "failed to read optional parameter flags";
)";
  }

  // For non-requiredParamsFirst attrs, read required params after flags.
  if (!attr.requiredParamsFirst) {
    for (const auto &param : attr.parameters) {
      if (!param.isOptional) {
        generateAttrParameterDeserialization(param, os, "  ",
                                             /*declareVariable=*/true);
      }
    }
  }

  // Deserialize optional parameters.
  unsigned optionalBitIndex = 0;
  for (const auto &param : attr.parameters) {
    if (param.isOptional) {
      // Optional param: conditionally deserialize based on flag.
      os << formatv(R"(  {0} {1};
  if (optionalFlags & (1ULL << {2})) {{
)",
                    param.cppType, param.name, optionalBitIndex++);
      generateAttrParameterDeserialization(param, os, "    ",
                                           /*declareVariable=*/false);
      os << "  }\n";
    }
  }

  // Build constructor arguments.
  std::string args;
  for (const auto &param : attr.parameters) {
    args += ", " + param.name;
  }

  // Use getChecked if the attr has a verifier, otherwise use get.
  if (attr.hasVerifier) {
    os << formatv(R"(
  result = {0}::getChecked(
      [&]() {{ return reader.emitError(); }, context{1});
  return success(result != nullptr);
}
)",
                  attr.qualifiedCppName, args);
  } else {
    os << formatv(R"(
  result = {0}::get(context{1});
  return success();
}
)",
                  attr.qualifiedCppName, args);
  }
}

void mlir::tblgen::generateAttrDeserializers(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  emitSourceFileHeader("Generated Attribute Deserialization Functions", os);
  os << R"(
// Auto-generated attribute deserializers.
// Format: [flags:varint if optional] [params...]
// For requiredParamsFirst attrs: [required params] [flags] [optional params]
// (Tag already consumed by dispatch)

)";
  for (const auto &attr : structure.serializableAttrs) {
    generateCudaTileAttrDeserializer(attr, os);
  }
}

//===----------------------------------------------------------------------===//
// Dispatch Generation
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateAttrSerializerDispatch(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  // Generate single dispatch function with writeTag parameter.
  os << R"(
// Auto-generated attribute serialization dispatch.
// If writeTag is true, writes the tag before data (for self-contained attrs).
// Returns success if attr was handled, failure otherwise.
LogicalResult writeCudaTileAttr(Operation *op, Attribute attr,
                                EncodingWriter &writer, bool writeTag) {
  return llvm::TypeSwitch<Attribute, LogicalResult>(attr)
)";

  for (const auto &serAttr : structure.serializableAttrs) {
    os << formatv(R"(      .Case<{0}>([&](auto a) {{
        if (writeTag)
          writer.writeVarInt(static_cast<uint8_t>(Bytecode::AttributeTag::{1}));
        return serialize{2}Data(op, a, writer);
      })
)",
                  serAttr.qualifiedCppName, serAttr.tagName, serAttr.attrName);
  }

  os << R"(      .Default([](Attribute) { return failure(); });
}
)";
}

void mlir::tblgen::generateSerializableAttrTypeTrait(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(
/// Helper type trait to check if T is one of the CudaTile serializable attrs
/// (attrs with auto-generated bytecode serialization/deserialization).
template <typename T>
struct is_cuda_tile_serializable_attr : std::disjunction<
)";

  llvm::interleave(
      structure.serializableAttrs,
      [&](const auto &attr) {
        os << "    std::is_same<T, " << attr.qualifiedCppName << ">";
      },
      [&] { os << ",\n"; });

  os << "> {};\n";
}

void mlir::tblgen::generateAttrInlineReaderDispatch(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(
/// Auto-generated compile-time dispatch for CudaTile serializable attributes.
/// Delegates to the appropriate parse*Data member function based on type T.
/// Guard calls with is_cuda_tile_serializable_attr<T>::value.
template <typename T>
LogicalResult parseCudaTileAttrInline(EncodingReader &reader,
                                      Attribute &result) {
  static_assert(is_cuda_tile_serializable_attr<T>::value,
                "T must be a CudaTile serializable attribute");
)";

  bool first = true;
  for (const auto &attr : structure.serializableAttrs) {
    os << formatv(R"(  {0}if constexpr (std::is_same_v<T, {1}>) {{
    return parse{2}Data(reader, result);
  })",
                  first ? "" : " else ", attr.qualifiedCppName, attr.attrName);
    first = false;
  }

  os << R"( else {
    return failure();
  }
}
)";
}

void mlir::tblgen::generateAttrDeserializerDispatch(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(
// Auto-generated attribute deserialization dispatch.
// Tag has already been read; this switches on it.
// Returns success if tag was handled, failure otherwise.
LogicalResult readCudaTileAttr(uint8_t tag, EncodingReader &reader,
                               Attribute &result) {
  switch (static_cast<Bytecode::AttributeTag>(tag)) {
)";

  for (const auto &serAttr : structure.serializableAttrs) {
    os << formatv(R"(  case Bytecode::AttributeTag::{0}:
    return parse{1}Data(reader, result);
)",
                  serAttr.tagName, serAttr.attrName);
  }

  os << R"(  default:
    return failure();
  }
}
)";
}

void mlir::tblgen::generateAttrReaderSwitchCases(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  os << R"(
// Auto-generated case labels for CudaTile serializable attributes.
// Include inside a switch on AttributeTag to get explicit cases for all
// CudaTile attrs. The handler block after these labels should delegate
// to AttrReader::readCudaTileAttr.
)";

  for (const auto &serAttr : structure.serializableAttrs) {
    os << formatv("case Bytecode::AttributeTag::{0}:\n", serAttr.tagName);
  }
}

//===----------------------------------------------------------------------===//
// Dependent Attribute Type Registration
//===----------------------------------------------------------------------===//

/// Returns true if the parameter kind references nested types or attrs that
/// must be registered with the type table before the enclosing attr is
/// serialized.
static bool needsAttrTypeRegistration(BytecodeAttrParameter::Kind kind) {
  return kind == BytecodeAttrParameter::Kind::TypeAttr ||
         kind == BytecodeAttrParameter::Kind::NestedAttr ||
         kind == BytecodeAttrParameter::Kind::PolymorphicNestedAttr ||
         kind == BytecodeAttrParameter::Kind::AttributeArray;
}

void mlir::tblgen::generateDependentAttrTypeRegistration(
    const BytecodeAttrStructure &structure, llvm::raw_ostream &os) {
  emitSourceFileHeader("Generated Dependent Attribute Type Registration", os);

  for (const auto &serAttr : structure.serializableAttrs) {
    if (!llvm::any_of(serAttr.parameters, [](const BytecodeAttrParameter &p) {
          return needsAttrTypeRegistration(p.kind);
        })) {
      continue;
    }

    os << formatv("if (auto concreteAttr = dyn_cast<{0}>(attr)) {{\n",
                  serAttr.qualifiedCppName);
    for (const auto &param : serAttr.parameters) {
      switch (param.kind) {
      case BytecodeAttrParameter::Kind::TypeAttr:
        os << formatv("  if (auto paramAttr = concreteAttr.{0}())\n"
                      "    getTypeIndex(paramAttr.getValue());\n",
                      param.accessorName);
        break;
      case BytecodeAttrParameter::Kind::NestedAttr:
      case BytecodeAttrParameter::Kind::PolymorphicNestedAttr:
        os << formatv(
            "  registerDependentAttributeTypes(concreteAttr.{0}());\n",
            param.accessorName);
        break;
      case BytecodeAttrParameter::Kind::AttributeArray:
        os << formatv("  for (Attribute elementAttr : concreteAttr.{0}())\n"
                      "    registerDependentAttributeTypes(elementAttr);\n",
                      param.accessorName);
        break;
      default:
        break;
      }
    }
    os << "  return;\n}\n";
  }
}
