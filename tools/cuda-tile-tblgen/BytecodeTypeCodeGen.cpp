//===- BytecodeTypeCodeGen.cpp ----------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BytecodeTypeCodeGen.h"

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Format.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "BytecodeGenUtilities.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Code Generation Templates.
//===----------------------------------------------------------------------===//

/// Template for integer type serializer function.
/// {0}: Integer type checks.
static const char *const integerSerializerTemplate = R"(
// Auto-generated integer type serialization with version checking
LogicalResult serializeIntegerType(IntegerType type, EncodingWriter &writer,
                                   const BytecodeWriterConfig &config) {{
  unsigned width = type.getWidth();
{0}
  return emitError(UnknownLoc::get(type.getContext()), "unsupported integer type width");
}
)";

/// Template for float type serializer function.
/// {0}: Float type checks.
static const char *const floatSerializerTemplate = R"(
// Auto-generated float type serialization with version checking
LogicalResult serializeFloatType(FloatType type, EncodingWriter &writer,
                                 const BytecodeWriterConfig &config) {{
{0}
  return emitError(UnknownLoc::get(type.getContext()), "unsupported float type: ") << type;
}
)";

/// Template for CudaTile type serializer function signature.
/// {0}: Type name, {1}: Qualified type name.
static const char *const cudaTileSerializerSignatureTemplate = R"(
// Writer for Type: {0}
LogicalResult serialize{0}({1} type,
                                   EncodingWriter &writer,
                                   const BytecodeWriterConfig &config) {{
)";

//===----------------------------------------------------------------------===//
// Parameter Serialization/Deserialization Templates.
//===----------------------------------------------------------------------===//

/// {0}: Getter call.
static const char *const serializeArrayTemplate =
    "writer.writeLEVarSize({0});\n";
static const char *const serializeDenseI32ArrayTemplate =
    "writer.writeLEVarSize({0}.asArrayRef());\n";
static const char *const serializeTypeTemplate = R"(
if (failed(writeTypeIndex({0}, writer)))
  return failure();
)";
static const char *const serializeTypeAttrTemplate = R"(
if (failed(writeTypeIndex({0}.getValue(), writer)))
  return failure();
)";
static const char *const serializeNestedAttrTemplate = R"(
if (failed(writeSelfContainedTypeAttribute({0}, writer)))
  return failure();
)";
/// Polymorphic nested-attr serialization template (self-contained encoding,
/// used when the param's declared C++ type is an AttrInterface).
/// {0}: getter call.
static const char *const serializePolymorphicNestedAttrTemplate = R"(
if (failed(writeSelfContainedTypeAttribute({0}, writer)))
  return failure();
)";
/// Template for AttributeArray serialization on the type side.
/// {0}: Getter call (size source), {1}: range expression to iterate.
static const char *const serializeAttrArrayTemplate = R"(
writer.writeVarInt({0}.size());
for (Attribute elementAttr : {1}) {{
  if (failed(writeSelfContainedTypeAttribute(elementAttr, writer)))
    return failure();
}
)";
/// Runtime template for OptionalEnum (includes both formats).
/// {0}: Getter call, {1}: Enum type, {2}: Runtime flag variable
static const char *const serializeOptionalEnumTemplate = R"(
if ({2}) {{
  if ({0})
    writer.writeVarInt(static_cast<std::underlying_type_t<{1}>>({0}.getValue()));
} else {{
  writer.writeByte({0} != nullptr);
  if ({0})
    writer.writeVarInt(static_cast<std::underlying_type_t<{1}>>({0}.getValue()));
}
)";
/// Templates for deserialization of original parameters.
/// {0}: Variable name.
static const char *const readArrayTemplate = R"(
if (failed(reader.readLEVarSize({0})))
 return reader.emitError() << "failed to read {0}";
)";

/// Template for uint32_t scalar serialization.
/// {0}: Getter call.
static const char *const serializeUInt32ScalarTemplate =
    "writer.writeVarInt(static_cast<uint64_t>({0}));\n";

/// Template for uint32_t scalar deserialization.
/// {0}: Variable name.
static const char *const readUInt32ScalarTemplate = R"(
uint64_t {0}_raw;
if (failed(reader.readVarInt({0}_raw)))
  return reader.emitError() << "failed to read {0}";
if ({0}_raw > UINT32_MAX)
  return reader.emitError() << "{0} value " << {0}_raw << " exceeds uint32_t range";
{0} = static_cast<uint32_t>({0}_raw);
)";

/// {0}: Variable name.
static const char *const readDenseI32ArrayTemplate = R"(
SmallVector<int32_t, 4> {0}_data;
if (failed(reader.readLEVarSize({0}_data)))
 return reader.emitError() << "failed to read {0} data";
{0} = DenseI32ArrayAttr::get(&context, {0}_data);
)";
static const char *const readTypeAttrTemplate = R"(
Type {0}_type = readAndGetType(reader);
if (!{0}_type)
  return reader.emitError() << "failed to get {0} type";
{0} = TypeAttr::get({0}_type);
)";
static const char *const readNestedAttrTemplate = R"(
if (failed(readNestedAttribute(reader, {0})))
  return failure();
)";
/// Polymorphic nested-attr deserialization template. Reads a generic
/// Attribute via tag dispatch then narrows to the declared interface type.
/// {0}: variable name. {1}: declared C++ interface type.
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
/// Template for AttributeArray deserialization on the type side.
/// Reads N self-contained attributes into an ArrayAttr stored in {0}.
/// {0}: Variable name.
static const char *const readAttrArrayTemplate = R"(
uint64_t {0}_size;
if (failed(reader.readVarInt({0}_size,
                             std::numeric_limits<uint32_t>::max() - 1)))
  return reader.emitError() << "failed to read {0} size";
SmallVector<Attribute, 4> {0}_storage;
{0}_storage.reserve({0}_size);
for (uint64_t i = 0; i < {0}_size; ++i) {{
  Attribute {0}_element;
  if (failed(readNestedAttribute(reader, {0}_element)))
    return reader.emitError() << "failed to read {0} element " << i;
  {0}_storage.push_back({0}_element);
}
{0} = ArrayAttr::get(&context, {0}_storage);
)";

/// {0}: Variable name.
static const char *const readGenericTypeTemplate = R"(
{0} = readAndGetType(reader);
if (!{0})
  return reader.emitError() << "failed to get {0} type";
)";

/// {0}: Variable name, {1}: C++ type.
static const char *const readSpecificTypeTemplate = R"(
Type {0}_generic = readAndGetType(reader);
if (!{0}_generic)
  return reader.emitError() << "failed to get {0} type";
{0} = ::mlir::dyn_cast<{1}>({0}_generic);
if (!{0})
  return reader.emitError() << "expected {1} but got " << {0}_generic;
)";
/// Runtime template for OptionalEnum deserialization (includes both formats).
/// {0}: Variable name, {1}: Runtime flag variable
static const char *const readOptionalEnumTemplate = R"(
if ({1}) {{
  if (failed(parseGenericEnumAttr(reader, context, {0})))
    return failure();
} else {{
  uint8_t presence;
  if (failed(reader.readLE(presence)))
    return reader.emitError() << "failed to read OptionalEnum presence flag";
  if (presence)
    if (failed(parseGenericEnumAttr(reader, context, {0})))
      return failure();
}
)";

//===----------------------------------------------------------------------===//
// Helper Functions.
//===----------------------------------------------------------------------===//

/// Get parameters in serialization order.
static SmallVector<BytecodeTypeParameter, 4>
getSerializationOrder(const CudaTileType &type) {
  if (type.needsReverseOrder) {
    return llvm::to_vector<4>(llvm::reverse(type.parameters));
  }
  return type.parameters;
}

static bool needsAttributeTypeRegistration(BytecodeTypeParameter::Kind kind) {
  return kind == BytecodeTypeParameter::Kind::TypeAttr ||
         kind == BytecodeTypeParameter::Kind::NestedAttr ||
         kind == BytecodeTypeParameter::Kind::PolymorphicNestedAttr ||
         kind == BytecodeTypeParameter::Kind::AttributeArray;
}

/// True for parameters whose value is an attribute that carries its own
/// bytecode version, so it must be scored when computing a type's min version.
static bool isAttributeVersionParameter(BytecodeTypeParameter::Kind kind) {
  return kind == BytecodeTypeParameter::Kind::OptionalEnum ||
         kind == BytecodeTypeParameter::Kind::NestedAttr ||
         kind == BytecodeTypeParameter::Kind::PolymorphicNestedAttr ||
         kind == BytecodeTypeParameter::Kind::AttributeArray;
}

static bool needsUnifiedBitfieldSerializationFlag(const CudaTileType &type) {
  return llvm::any_of(
      getSerializationOrder(type), [](const BytecodeTypeParameter &param) {
        return param.kind == BytecodeTypeParameter::Kind::OptionalEnum;
      });
}

/// Check if version string is >= 13.3 (unified bitfield format).
/// Types introduced at 13.3+ always use unified format.
static bool isUnifiedBitfieldVersion(StringRef version) {
  auto [majorStr, minorStr] = parseVersion(version);
  unsigned major, minor;
  if (majorStr.getAsInteger(10, major) || minorStr.getAsInteger(10, minor)) {
    PrintFatalError("invalid version string: " + version.str());
  }
  return (major > 13) || (major == 13 && minor >= 3);
}

/// Replace TableGen placeholders with C++ code.
static std::string replaceTableGenPlaceholders(StringRef code,
                                               StringRef ctxtVar) {
  FmtContext ctx;
  ctx.addSubst("_ctxt", ctxtVar);
  return tgfmt(code, &ctx);
}

/// Generate version check with proper indentation.
static std::string
generateVersionCheck(unsigned indent, StringRef version, StringRef typeName,
                     StringRef versionVar = "config.bytecodeVersion",
                     StringRef contextExpr = "type.getContext()",
                     StringRef errorSuffix = "targeting ") {
  auto [majorStr, minorStr] = parseVersion(version);
  std::string indentStr(indent, ' ');
  return formatv("{0}if ({1} < *BytecodeVersion::fromVersion({2}, {3}, 0))\n"
                 "{0}  return ::emitError(UnknownLoc::get({4}))\n"
                 "{0}         << \"type '{5}' requires bytecode version {6}+, "
                 "{7}\" << {1}.toString();\n",
                 indentStr, versionVar, majorStr, minorStr, contextExpr,
                 typeName, version, errorSuffix)
      .str();
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Tag Enum.
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateTypeTagEnum(const BytecodeTypeStructure &structure,
                                       raw_ostream &os) {
  emitSourceFileHeader("Generated TypeTag Enum", os);
  os << "/// FROZEN at current assignments for backward compatibility.\n"
     << "/// WARNING: NEVER CHANGE THESE VALUES - they must remain stable.\n"
     << "enum class TypeTag : uint8_t {\n";
  // Generate all type tags.
  for (const auto &[enumName, tagValue] : structure.allTypeTags) {
    os << "  " << enumName << " = " << tagValue << ",\n";
  }
  os << "};\n";
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Serialization.
//===----------------------------------------------------------------------===//

static void generateParameterSerialization(const BytecodeTypeParameter &param,
                                           raw_ostream &os,
                                           StringRef indent = "",
                                           StringRef runtimeFlag = "") {
  std::string getterCall = "type." + param.accessorName + "()";
  std::string attributeArrayRange =
      (param.cppType.find("ArrayAttr") != std::string::npos ||
       param.cppStorageType.find("ArrayAttr") != std::string::npos)
          ? getterCall + ".getValue()"
          : getterCall;
  mlir::raw_indented_ostream ios(os);

  switch (param.kind) {
  case BytecodeTypeParameter::Kind::TypeAttr:
    ios.printReindented(formatv(serializeTypeAttrTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::NestedAttr:
    ios.printReindented(formatv(serializeNestedAttrTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::PolymorphicNestedAttr:
    ios.printReindented(
        formatv(serializePolymorphicNestedAttrTemplate, getterCall).str(),
        indent);
    break;
  case BytecodeTypeParameter::Kind::AttributeArray:
    ios.printReindented(
        formatv(serializeAttrArrayTemplate, getterCall, attributeArrayRange)
            .str(),
        indent);
    break;
  case BytecodeTypeParameter::Kind::Int64Array:
  case BytecodeTypeParameter::Kind::Int32Array:
    ios.printReindented(formatv(serializeArrayTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::UInt32Scalar:
    ios.printReindented(
        formatv(serializeUInt32ScalarTemplate, getterCall).str(), indent);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    ios.printReindented(
        formatv(serializeDenseI32ArrayTemplate, getterCall).str(), indent);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
  case BytecodeTypeParameter::Kind::SpecificType:
    ios.printReindented(formatv(serializeTypeTemplate, getterCall).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    ios.printReindented(formatv(serializeOptionalEnumTemplate, getterCall,
                                param.enumTypeName, runtimeFlag)
                            .str(),
                        indent);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Parameter Deserialization.
//===----------------------------------------------------------------------===//

static void generateParameterDeserialization(const BytecodeTypeParameter &param,
                                             raw_ostream &os,
                                             StringRef indent = "",
                                             bool declareVariable = true,
                                             StringRef runtimeFlag = "") {
  mlir::raw_indented_ostream ios(os);

  switch (param.kind) {
  case BytecodeTypeParameter::Kind::TypeAttr:
    if (declareVariable) {
      os << indent << formatv("TypeAttr {0};\n", param.name);
    }
    ios.printReindented(formatv(readTypeAttrTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::NestedAttr:
    if (declareVariable) {
      os << indent << formatv("Attribute {0};\n", param.name);
    }
    ios.printReindented(formatv(readNestedAttrTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::PolymorphicNestedAttr:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, param.name);
    }
    ios.printReindented(
        formatv(readPolymorphicNestedAttrTemplate, param.name, param.cppType)
            .str(),
        indent);
    break;
  case BytecodeTypeParameter::Kind::AttributeArray:
    if (declareVariable) {
      os << indent << formatv("ArrayAttr {0};\n", param.name);
    }
    ios.printReindented(formatv(readAttrArrayTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::Int64Array:
    if (declareVariable) {
      os << indent << formatv("SmallVector<int64_t, 4> {0};\n", param.name);
    }
    ios.printReindented(formatv(readArrayTemplate, param.name).str(), indent);
    break;
  case BytecodeTypeParameter::Kind::Int32Array:
    if (declareVariable) {
      os << indent << formatv("SmallVector<int32_t, 4> {0};\n", param.name);
    }
    ios.printReindented(formatv(readArrayTemplate, param.name).str(), indent);
    break;
  case BytecodeTypeParameter::Kind::UInt32Scalar:
    if (declareVariable) {
      os << indent << formatv("uint32_t {0};\n", param.name);
    }
    ios.printReindented(formatv(readUInt32ScalarTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::DenseI32Array:
    if (declareVariable) {
      os << indent << formatv("DenseI32ArrayAttr {0};\n", param.name);
    }
    ios.printReindented(formatv(readDenseI32ArrayTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::GenericType:
    if (declareVariable) {
      os << indent << formatv("Type {0};\n", param.name);
    }
    ios.printReindented(formatv(readGenericTypeTemplate, param.name).str(),
                        indent);
    break;
  case BytecodeTypeParameter::Kind::SpecificType:
    if (declareVariable) {
      os << indent << formatv("{0} {1};\n", param.cppType, param.name);
    }
    ios.printReindented(
        formatv(readSpecificTypeTemplate, param.name, param.cppType).str(),
        indent);
    break;
  case BytecodeTypeParameter::Kind::OptionalEnum:
    if (declareVariable) {
      os << indent << formatv("{1} {0};\n", param.name, param.cppType);
    }
    ios.printReindented(
        formatv(readOptionalEnumTemplate, param.name, runtimeFlag).str(),
        indent);
    break;
  default:
    llvm::PrintFatalError("Unsupported parameter kind in code generation");
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Built-in Type Serializers.
//===----------------------------------------------------------------------===//

/// Generate serializers for all built-in types.
static void
generateBuiltinTypeSerializers(const BytecodeTypeStructure &structure,
                               raw_ostream &os) {
  std::string intChecks, floatChecks;

  for (const auto &bt : structure.builtinSerializableTypes) {
    std::string condition =
        bt.isInteger() ? formatv("width == {0}", bt.integerBitWidth).str()
                       : formatv("isa<{0}>(type)", bt.floatMlirTypeName).str();
    std::string check =
        formatv(R"(
  if ({0}) {{
{1}
    writer.writeVarInt(Bytecode::TypeTag::{2});
    return success();
  })",
                condition,
                generateVersionCheck(4, bt.sinceVersion, bt.enumName),
                bt.enumName)
            .str();
    if (bt.isInteger()) {
      intChecks += check;
    }
    if (bt.isFloat()) {
      floatChecks += check;
    }
  }
  if (!intChecks.empty()) {
    os << formatv(integerSerializerTemplate, intChecks);
  }
  if (!floatChecks.empty()) {
    os << formatv(floatSerializerTemplate, floatChecks);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Optional Parameter Flags (Writer).
//===----------------------------------------------------------------------===//
//
// Optional Parameter Handling:
//
// Types can have optional parameters (OptionalParameter<T>) that may be null.
// These use a bitfield-based encoding for efficient serialization.
//
// Format Evolution:
//   Version <13.3 (Legacy):
//     - OptionalEnum: Uses inline flag byte per parameter
//     - Optional Type: Not supported as it was added in later versions.
//
//   Version ≥13.3 (Unified Bitfield):
//     - ALL optional params (Type, Enum, etc.): Single unified bitfield
//     - No inline flags - bitfield indicates presence
//
// Bitfield Layout:
//   - Bit 0: First optional param (in serialization order)
//   - Bit N: Nth optional param
//   - Bit set (1) = present, Bit clear (0) = null/absent
//
// Version Checking:
//   Flags written if:
//     1. config.bytecodeVersion >= firstOptionalParamVersion (type supports it)
//     2. config.bytecodeVersion >= 13.3 (unified format exists)
//   This allows targeting older versions for backward compatibility.
//
//===----------------------------------------------------------------------===//

/// Generates flags field serialization for optional type parameters.
static void generateOptionalParamFlags(const CudaTileType &type,
                                       raw_ostream &os) {
  if (!type.hasOptionalTypeParams) {
    return;
  }

  if (needsUnifiedBitfieldSerializationFlag(type)) {
    if (type.skipVersionCheck) {
      os << R"(
  bool useUnifiedBitfield = true;
  (void)useUnifiedBitfield;
)";
    } else {
      os << R"(
  bool useUnifiedBitfield = config.bytecodeVersion >= BytecodeVersion::kUnifiedBitfieldVersion;
  (void)useUnifiedBitfield;
)";
    }
  }

  os << R"(  uint64_t optionalFlags = 0;
)";

  // Build flags for all optional params.
  unsigned bitIndex = 0;
  for (const auto &param : getSerializationOrder(type)) {
    if (param.usesOptionalTypeFlags) {
      os << formatv(R"(  if (type.{0}()) optionalFlags |= (1ULL << {1});
)",
                    param.accessorName, bitIndex++);
    }
  }

  // Write flags.
  if (type.skipVersionCheck) {
    os << "  writer.writeVarInt(optionalFlags);\n";
  } else {
    std::string minOptionalVersionStr = type.firstOptionalTypeParamVersion;
    bool needsVersionCheck = !minOptionalVersionStr.empty() &&
                             (minOptionalVersionStr != type.sinceVersion);
    // Types >= 13.3 don't need runtime >= 13.3 check.
    bool typeIsUnified = isUnifiedBitfieldVersion(type.sinceVersion);

    if (needsVersionCheck) {
      auto [majorStr, minorStr] = parseVersion(minOptionalVersionStr);
      if (typeIsUnified) {
        // Type >= 13.3: only check >= firstOptionalVersion.
        os << formatv(R"(  
  if (config.bytecodeVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0))
    writer.writeVarInt(optionalFlags);
)",
                      majorStr, minorStr);
      } else {
        // Type < 13.3: check both versions.
        os << formatv(R"(  
  if (config.bytecodeVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0) &&
      config.bytecodeVersion >= BytecodeVersion::kUnifiedBitfieldVersion)
    writer.writeVarInt(optionalFlags);
)",
                      majorStr, minorStr);
      }
    } else {
      if (typeIsUnified) {
        // Type >= 13.3: no check needed.
        os << "  writer.writeVarInt(optionalFlags);\n";
      } else {
        // Type < 13.3: check >= 13.3.
        os << R"(  
  if (config.bytecodeVersion >= BytecodeVersion::kUnifiedBitfieldVersion)
    writer.writeVarInt(optionalFlags);
)";
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Serializers.
//===----------------------------------------------------------------------===//
//
// Generates serialization function for each CudaTile type.
// For types with optional parameters, uses unified bitfield (13.3+) or
// legacy inline flags (<13.3). See "Optional Parameter Flags" section above.
//
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeSerializer(const CudaTileType &type,
                                           raw_ostream &os) {
  // Function signature
  os << formatv(cudaTileSerializerSignatureTemplate, type.typeName,
                type.qualifiedTypeName);

  if (!type.skipVersionCheck) {
    os << generateVersionCheck(2, type.sinceVersion, type.typeName);
  }

  // Write type tag.
  os << "  writer.writeVarInt(Bytecode::TypeTag::" << type.typeName << ");\n";

  // Declare format flag and write flags only if type has optional params.
  generateOptionalParamFlags(type, os);

  // Serialize parameters conditionally based on target bytecode version.
  for (const auto &param : getSerializationOrder(type)) {
    bool isEvolved =
        !type.skipVersionCheck && param.sinceVersion != type.sinceVersion;

    // Original parameters - always serialize. Gate optional params on
    // their presence flag to avoid writing null values, except for
    // OptionalEnum: its template emits the legacy presence byte itself
    // (0 for absent, 1 for present), so wrapping it would elide the
    // byte the pre-13.3 reader unconditionally consumes.
    if (!isEvolved) {
      bool isOptionalEnum =
          param.kind == BytecodeTypeParameter::Kind::OptionalEnum;
      if (param.usesOptionalTypeFlags && !isOptionalEnum) {
        std::string getterCall = "type." + param.accessorName + "()";
        os << formatv("  if ({0}) {{\n", getterCall);
        generateParameterSerialization(param, os, "    ", "useUnifiedBitfield");
        os << "  }\n";
      } else {
        generateParameterSerialization(param, os, "  ", "useUnifiedBitfield");
      }
      continue;
    }

    // Evolved parameters - version-guarded with validation.
    auto [majorStr, minorStr] = parseVersion(param.sinceVersion);
    std::string getterCall = "type." + param.accessorName + "()";
    std::string defaultValue =
        replaceTableGenPlaceholders(param.defaultValue, "type.getContext()");

    // Validate: null check for optional, equality for non-optional.
    std::string validationCheck =
        param.usesOptionalTypeFlags
            ? formatv("if ({0})", getterCall).str()
            : formatv("if ({0} != {1})", getterCall, defaultValue).str();

    os << formatv(R"(
  if (config.bytecodeVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0)) {{
)",
                  majorStr, minorStr);

    // Serialize parameter. OptionalEnum is special-cased as above: its
    // template emits the legacy presence byte itself, so do not wrap.
    bool isOptionalEnum =
        param.kind == BytecodeTypeParameter::Kind::OptionalEnum;
    if (param.usesOptionalTypeFlags && !isOptionalEnum) {
      os << formatv("    if ({0}) {{\n", getterCall);
      generateParameterSerialization(param, os, "      ", "useUnifiedBitfield");
      os << "    }\n";
    } else {
      generateParameterSerialization(param, os, "    ", "useUnifiedBitfield");
    }

    os << formatv(R"(  } else {{
      // Validate: parameter must be null/equal default when targeting older version.
      {0}
      return emitError(UnknownLoc::get(type.getContext()),
                   "parameter '{1}' requires bytecode version {2}+, but targeting ") << config.bytecodeVersion.toString();
  }
)",
                  validationCheck, param.name, param.sinceVersion);
  }

  os << "  return success();\n}\n\n";
}

void mlir::tblgen::generateTypeSerializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Serialization Functions", os);
  generateBuiltinTypeSerializers(structure, os);
  for (const auto &type : structure.cudaTileTypes) {
    generateCudaTileTypeSerializer(type, os);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Built-in Type Deserializers.
//===----------------------------------------------------------------------===//

/// Generate deserializers for all built-in types.
static void
generateBuiltinTypeDeserializers(const BytecodeTypeStructure &structure,
                                 raw_ostream &os) {
  std::string intChecks, floatChecks;

  for (const auto &bt : structure.builtinSerializableTypes) {
    std::string typeCreation =
        bt.isInteger()
            ? formatv("IntegerType::get(&context, {0})", bt.integerBitWidth)
                  .str()
            : formatv("{0}::get(&context)", bt.floatMlirTypeName).str();

    // Generate version check.
    std::string versionCheck =
        generateVersionCheck(4, bt.sinceVersion, bt.enumName, "fileVersion",
                             "&context", "file version is ");

    std::string check = formatv(R"(
  if (typeTag == {0}) {{
{1}    result = {2};
    return success();
  })",
                                bt.typeTagValue, versionCheck, typeCreation)
                            .str();

    if (bt.isInteger()) {
      intChecks += check;
    }
    if (bt.isFloat()) {
      floatChecks += check;
    }
  }

  if (!intChecks.empty()) {
    os << formatv(R"(// Auto-generated integer type deserialization
LogicalResult parseIntegerType(uint8_t typeTag, Type &result, MLIRContext &context,
                               const BytecodeVersion &fileVersion) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "invalid integer type tag: " << static_cast<int>(typeTag);
})",
                  intChecks);
  }

  if (!floatChecks.empty()) {
    os << formatv(R"(// Auto-generated float type deserialization
LogicalResult parseFloatType(uint8_t typeTag, Type &result, MLIRContext &context,
                             const BytecodeVersion &fileVersion) {{
{0}  
  return ::emitError(UnknownLoc::get(&context)) << "unsupported float type tag: " << static_cast<int>(typeTag);
})",
                  floatChecks);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Optional Parameter Flags (Reader).
//===----------------------------------------------------------------------===//
//
// Reads the optional parameter bitfield written by the writer.
// See "Optional Parameter Flags (Writer)" section above for format details.
//
// Reading Strategy:
//   - Determine format from fileVersion (≥13.3 = unified, <13.3 = legacy)
//   - Read bitfield if: fileVersion >= firstOptionalParamVersion && unified
//   - For legacy format: OptionalEnum uses inline flags (read by templates)
//
//===----------------------------------------------------------------------===//

/// Generates flags field deserialization for optional type parameters.
static void generateOptionalParamFlagsReader(const CudaTileType &type,
                                             raw_ostream &os) {
  if (!type.hasOptionalTypeParams) {
    return;
  }

  if (type.skipVersionCheck) {
    os << R"(
  bool useUnifiedBitfield = true;
  (void)useUnifiedBitfield;
  uint64_t optionalFlags = 0;
  if (failed(reader.readVarInt(optionalFlags)))
    return reader.emitError() << "failed to read optional parameter flags";
)";
  } else {
    // Types >= 13.3 don't need runtime >= 13.3 check.
    bool typeIsUnified = isUnifiedBitfieldVersion(type.sinceVersion);
    std::string minOptionalVersionStr = type.firstOptionalTypeParamVersion;
    auto [majorStr, minorStr] = parseVersion(minOptionalVersionStr);

    if (typeIsUnified) {
      // Type >= 13.3: useUnifiedBitfield is always true.
      os << formatv(R"(
  bool useUnifiedBitfield = true;
  (void)useUnifiedBitfield;
  uint64_t optionalFlags = 0;
  if (fileVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0)) {{
    if (failed(reader.readVarInt(optionalFlags)))
      return reader.emitError() << "failed to read optional parameter flags";
  }
)",
                    majorStr, minorStr);
    } else {
      // Type < 13.3: need runtime check for unified format.
      os << R"(
  bool useUnifiedBitfield = fileVersion >= BytecodeVersion::kUnifiedBitfieldVersion;
  (void)useUnifiedBitfield;
)";
      os << formatv(R"(
  uint64_t optionalFlags = 0;
  if (fileVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0) && useUnifiedBitfield) {{
    if (failed(reader.readVarInt(optionalFlags)))
      return reader.emitError() << "failed to read optional parameter flags";
  }
)",
                    majorStr, minorStr);
    }
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Type Deserializers.
//===----------------------------------------------------------------------===//
//
// Generates deserialization function for each CudaTile type.
// Reads optional parameter bitfield (13.3+) or inline flags (<13.3).
// See "Optional Parameter Flags" sections above for format details.
//
//===----------------------------------------------------------------------===//

static void generateCudaTileTypeDeserializer(const CudaTileType &type,
                                             raw_ostream &os) {
  os << formatv(R"(// Reader for Type: {0}
LogicalResult parse{0}(EncodingReader &reader, Type &result) {{
)",
                type.typeName);

  // Version check for the type itself.
  if (!type.skipVersionCheck) {
    os << generateVersionCheck(2, type.sinceVersion, type.typeName,
                               "fileVersion", "&context", "file version is ");
  }

  // Declare format flag and read flags only if type has optional params.
  generateOptionalParamFlagsReader(type, os);

  // Deserialize parameters conditionally based on file bytecode version.
  unsigned optionalBitIndex = 0;
  for (const auto &param : getSerializationOrder(type)) {
    bool isEvolved =
        !type.skipVersionCheck && param.sinceVersion != type.sinceVersion;

    // Original parameters - always deserialize.
    if (!isEvolved) {
      // Optional params: template chooses format based on useUnifiedBitfield.
      if (param.usesOptionalTypeFlags) {
        os << formatv(R"(  {0} {1};
  if (!useUnifiedBitfield || (optionalFlags & (1ULL << {2}))) {{
)",
                      param.cppType, param.name, optionalBitIndex++);
        generateParameterDeserialization(
            param, os, "    ", /*declareVariable=*/false, "useUnifiedBitfield");
        os << R"(  }
)";
      } else {
        generateParameterDeserialization(
            param, os, "  ", /*declareVariable=*/true, "useUnifiedBitfield");
      }
      continue;
    }

    // Evolved parameters - version-guarded deserialization.
    auto [majorStr, minorStr] = parseVersion(param.sinceVersion);
    std::string defaultValue =
        replaceTableGenPlaceholders(param.defaultValue, "&context");

    // Optional params: conditionally read based on flag bit (template handles
    // runtime format).
    if (param.usesOptionalTypeFlags) {
      os << formatv(R"(  {0} {1};
  if (optionalFlags & (1ULL << {2})) {{
)",
                    param.cppType, param.name, optionalBitIndex++);
      generateParameterDeserialization(
          param, os, "    ", /*declareVariable=*/false, "useUnifiedBitfield");
      os << "  }\n";
      continue;
    }

    // Non-optional params: initialize with default, conditionally override.
    os << formatv("  {0} {1} = {2};\n", param.cppType, param.name,
                  defaultValue);
    os << formatv(R"(
  if (fileVersion >= *BytecodeVersion::fromVersion({0}, {1}, 0)) {{
)",
                  majorStr, minorStr);
    generateParameterDeserialization(
        param, os, "    ", /*declareVariable=*/false, "useUnifiedBitfield");
    os << "  }\n";
  }

  // Build constructor arguments.
  std::string args;
  for (const auto &param : type.parameters) {
    StringRef paramTypeRef(param.cppStorageType);
    if (paramTypeRef.contains("SmallVector<int64_t>")) {
      args += ", ArrayRef<int64_t>(" + param.name + ")";
    } else if (paramTypeRef.contains("SmallVector<int32_t>")) {
      args += ", ArrayRef<int32_t>(" + param.name + ")";
    } else {
      args += ", " + param.name;
    }
  }

  os << formatv(R"(  
  result = {0}::getChecked(
      [&]() {{ return reader.emitError(); }, &context{1});
  return success(result);
}
)",
                type.qualifiedTypeName, args);
}

void mlir::tblgen::generateTypeDeserializers(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Functions", os);
  generateBuiltinTypeDeserializers(structure, os);
  for (const auto &type : structure.cudaTileTypes) {
    generateCudaTileTypeDeserializer(type, os);
  }
}

//===----------------------------------------------------------------------===//
// C++ Generator - Dispatch.
//===----------------------------------------------------------------------===//

void mlir::tblgen::generateSerializerDispatch(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Serialization Dispatch", os);

  os << R"(return TypeSwitch<Type, LogicalResult>(type)
)";

  // Built-in types.
  if (llvm::any_of(structure.builtinSerializableTypes,
                   [](auto &t) { return t.isInteger(); })) {
    os << R"(    .Case<IntegerType>([&](auto concreteType) {
      return serializeIntegerType(concreteType, writer, config);
    })
)";
  }

  if (llvm::any_of(structure.builtinSerializableTypes,
                   [](auto &t) { return t.isFloat(); })) {
    os << R"(    .Case<FloatType>([&](auto concreteType) {
      return serializeFloatType(concreteType, writer, config);
    })
)";
  }

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes) {
    os << formatv(R"(    .Case<{0}>([&](auto concreteType) {{
      return serialize{1}(concreteType, writer, config);
    })
)",
                  type.qualifiedTypeName, type.typeName);
  }

  // FunctionType and default case.
  os << R"(    .Case<FunctionType>([&](auto concreteType) {
      return serializeFunctionType(concreteType, writer);
    })
    .Default([&](Type) {
      return emitError(UnknownLoc::get(type.getContext()),
                       "unsupported type in bytecode writer");
    });
)";
}

void mlir::tblgen::generateDependentTypeRegistration(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Dependent Type Registration", os);

  for (const auto &type : structure.cudaTileTypes) {
    // Check if type has nested type-bearing parameters that need registration.
    if (!llvm::any_of(type.parameters, [](const BytecodeTypeParameter &p) {
          return isTypeParameter(p.kind) ||
                 needsAttributeTypeRegistration(p.kind);
        })) {
      continue;
    }

    // Generate registration for this type.
    os << formatv("if (auto concreteType = dyn_cast<{0}>(type)) {{\n",
                  type.qualifiedTypeName);
    for (const auto &param : type.parameters) {
      if (isTypeParameter(param.kind)) {
        os << formatv(R"(  if (auto paramType = concreteType.{0}())
    getTypeIndex(paramType);
)",
                      param.accessorName);
        continue;
      }

      if (needsAttributeTypeRegistration(param.kind)) {
        os << formatv(R"(  registerDependentAttributeTypes(concreteType.{0}());
)",
                      param.accessorName);
      }
    }

    os << "  return;\n}\n";
  }
}

void mlir::tblgen::generateDeserializerDispatch(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Deserialization Dispatch", os);

  os << R"(switch (static_cast<TypeTag>(typeTag)) {
)";

  // Built-in types.
  for (const auto &builtinType : structure.builtinSerializableTypes) {
    os << "case Bytecode::TypeTag::" << builtinType.enumName << ":\n";
    if (builtinType.isInteger()) {
      os << "  return parseIntegerType(typeTag, result, context, "
            "fileVersion);\n";
    } else if (builtinType.isFloat()) {
      os << "  return parseFloatType(typeTag, result, context, fileVersion);\n";
    }
  }

  // CudaTile types.
  for (const auto &type : structure.cudaTileTypes) {
    os << formatv(R"(case Bytecode::TypeTag::{0}:
  return parse{0}(reader, result);
)",
                  type.typeName);
  }

  // FunctionType and default.
  os << R"(case Bytecode::TypeTag::FunctionType:
  return parseFunctionType(reader, result);
default:
  return ::emitError(UnknownLoc::get(&context))
         << "unknown type tag: " << static_cast<int>(typeTag);
}
)";
}

void mlir::tblgen::generateTypeVersionMap(
    const BytecodeTypeStructure &structure, raw_ostream &os) {
  emitSourceFileHeader("Generated Type Version Map", os);

  // Build integer switch cases and float .Case<> entries.
  // Skip minimum version entries - they don't raise the min version.
  std::string integerCasesStr;
  std::string floatCasesStr;
  llvm::raw_string_ostream integerCases(integerCasesStr);
  llvm::raw_string_ostream floatCases(floatCasesStr);
  for (const auto &bt : structure.builtinSerializableTypes) {
    auto [majorStr, minorStr] = parseVersion(bt.sinceVersion);
    if (isMinimumVersion(majorStr, minorStr)) {
      continue;
    }

    if (bt.isInteger()) {
      integerCases << formatv(R"(        case {0}:
          return BytecodeVersion::fromVersion({1}, {2}, 0);
)",
                              bt.integerBitWidth, majorStr, minorStr);
    } else if (bt.isFloat()) {
      floatCases << formatv(R"(      .Case<mlir::{0}>([](auto) {{
        return BytecodeVersion::fromVersion({1}, {2}, 0);
      })
)",
                            bt.floatMlirTypeName, majorStr, minorStr);
    }
  }

  os << R"(namespace mlir::cuda_tile::detail {

/// Returns the minimum bytecode version required for an MLIR Type.
/// Returns nullopt for unsupported types or those at minimum version.
inline std::optional<BytecodeVersion>
getTypeMinVersion(mlir::Type type) {
  return llvm::TypeSwitch<Type, std::optional<BytecodeVersion>>(type)
)";

  // Only generate IntegerType case if we have non-minimum version integers.
  if (!integerCasesStr.empty()) {
    os << R"(      .Case<mlir::IntegerType>([](mlir::IntegerType intType)
                              -> std::optional<BytecodeVersion> {
        switch (intType.getWidth()) {
)" << integerCasesStr
       << R"(        default:
          return std::nullopt;
        }
      })
)";
  }

  // Only generate float cases if we have non-minimum version floats.
  if (!floatCasesStr.empty()) {
    os << floatCasesStr;
  }

  // Single pass: generate getTypeMinVersion cases and accumulate
  // getTypeMinVersionRecursive cases.
  std::string recursiveCasesStr;
  llvm::raw_string_ostream recursiveCases(recursiveCasesStr);

  for (const auto &type : structure.cudaTileTypes) {
    // For getTypeMinVersion: non-minimum version types get a .Case entry.
    if (!type.skipVersionCheck) {
      auto [majorStr, minorStr] = parseVersion(type.sinceVersion);
      if (!isMinimumVersion(majorStr, minorStr)) {
        os << formatv(R"(      .Case<{0}>([](auto) {{
        return BytecodeVersion::fromVersion({1}, {2}, 0);
      })
)",
                      type.qualifiedTypeName, majorStr, minorStr);
      }
    }

    // For getTypeMinVersionRecursive: types with nested Type parameters or
    // attribute-valued parameters need recursive processing.
    if (llvm::any_of(type.parameters, [](const BytecodeTypeParameter &p) {
          return isTypeParameter(p.kind) || isAttributeVersionParameter(p.kind);
        })) {
      recursiveCases << formatv(R"(  if (auto t = dyn_cast<{0}>(type)) {{
)",
                                type.qualifiedTypeName);
      for (const auto &param : type.parameters) {
        if (isTypeParameter(param.kind)) {
          recursiveCases << formatv(R"(    if (auto paramType = t.{0}())
      updateMax(getTypeMinVersionRecursive(paramType));
)",
                                    param.accessorName);
        } else if (isAttributeVersionParameter(param.kind)) {
          if (param.kind == BytecodeTypeParameter::Kind::AttributeArray) {
            recursiveCases << formatv(R"(    for (Attribute paramAttr : t.{0}())
      updateMax(getAttrMinVersionRecursive(paramAttr));
)",
                                      param.accessorName);
          } else {
            recursiveCases << formatv(R"(    if (auto paramAttr = t.{0}())
      updateMax(getAttrMinVersionRecursive(paramAttr));
)",
                                      param.accessorName);
          }
        }
      }
      recursiveCases << "  }\n";
    }
  }

  os << R"(      .Default([](Type) -> std::optional<BytecodeVersion> {
        return std::nullopt;
      });
}

/// Process nested types recursively and return the maximum version required.
/// This handles composite types like TileType, TensorViewType, etc.
inline std::optional<BytecodeVersion>
getTypeMinVersionRecursive(Type type) {
  std::optional<BytecodeVersion> maxVersion;

  auto updateMax = [&](std::optional<BytecodeVersion> v) {
    if (v && (!maxVersion || *v > *maxVersion))
      maxVersion = v;
  };

  // Get this type's version.
  updateMax(getTypeMinVersion(type));

  // FunctionType: has input and result types that need recursive processing.
  if (auto funcType = dyn_cast<FunctionType>(type)) {
    for (Type input : funcType.getInputs())
      updateMax(getTypeMinVersionRecursive(input));
    for (Type result : funcType.getResults())
      updateMax(getTypeMinVersionRecursive(result));
  }

  // Process CudaTile types with nested Type parameters.
)" << recursiveCasesStr
     << R"(
  return maxVersion;
}

} // namespace mlir::cuda_tile::detail
)";
}
