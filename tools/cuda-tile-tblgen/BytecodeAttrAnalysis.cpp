//===- BytecodeAttrAnalysis.cpp ---------------------------------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// BytecodeAttrAnalysis.cpp - Attribute Bytecode Analysis Implementation
//
//===----------------------------------------------------------------------===//

#include "BytecodeAttrAnalysis.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

/// Extracts the attribute tag name from a TableGen definition name.
static StringRef extractAttrTagName(StringRef defName) {
  defName.consume_front("CudaTile_");
  defName.consume_back("Attr");
  return defName;
}

//===----------------------------------------------------------------------===//
// Analysis Implementation
//===----------------------------------------------------------------------===//

BytecodeAttrStructure
mlir::tblgen::analyzeBytecodeAttrs(const RecordKeeper &records) {
  BytecodeAttrStructure structure;

  // Build map of MLIR built-in attrs from CudaTileAttrAlias records.
  // These provide version info for MLIR built-in attributes we serialize.
  StringMap<StringRef> aliasVersions;
  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrAlias")) {
    StringRef attrName = record->getValueAsString("attrName");
    StringRef version = record->getValueAsString("sinceVersion");
    if (attrName.empty())
      PrintFatalError(record->getLoc(), "CudaTileAttrAlias '" +
                                            record->getName().str() +
                                            "' has empty 'attrName'");
    if (version.empty())
      PrintFatalError(record->getLoc(), "CudaTileAttrAlias '" +
                                            record->getName().str() +
                                            "' has empty 'sinceVersion'");
    aliasVersions[attrName] = version;
  }

  // Build map of CudaTileAttrDef records.
  struct AttrDefInfo {
    StringRef sinceVersion;
    bool skipsVersionCheck;
  };
  StringMap<AttrDefInfo> attrDefInfo;
  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrDef")) {
    bool skips = false;
    StringRef version;
    if (record->getValue("sinceVersion") &&
        !record->isValueUnset("sinceVersion")) {
      version = record->getValueAsString("sinceVersion");
    }

    if (!skips && version.empty())
      PrintFatalError(record->getLoc(),
                      "CudaTileAttrDef '" + record->getName().str() +
                          "' is missing or has empty 'sinceVersion'");

    attrDefInfo[extractAttrTagName(record->getName())] = {version, skips};
  }

  // Process BytecodeAttrTag records.
  for (const Record *record :
       records.getAllDerivedDefinitions("BytecodeAttrTag")) {
    StringRef attrName = record->getValueAsString("cppAttrName");

    BytecodeAttr attr;
    attr.attrName = attrName.str();
    attr.tagValue = record->getValueAsInt("attrTagValue");

    // Look up version from CudaTileAttrAlias or CudaTileAttrDef.
    // Every BytecodeAttrTag must have version info from one of these sources.
    if (auto it = aliasVersions.find(attrName); it != aliasVersions.end()) {
      attr.sinceVersion = it->second.str();
    } else if (auto it = attrDefInfo.find(attrName); it != attrDefInfo.end()) {
      attr.sinceVersion = it->second.sinceVersion.str();
      attr.skipsVersionCheck = it->second.skipsVersionCheck;
    } else {
      PrintFatalError(record->getLoc(),
                      "BytecodeAttrTag '" + attrName.str() +
                          "' has no matching CudaTileAttrAlias or "
                          "CudaTileAttrDef for version info");
    }

    structure.bytecodeAttrs.emplace_back(std::move(attr));
  }

  // Process enum attributes.
  auto processEnumDefs = [&](StringRef baseClass) {
    for (const Record *record : records.getAllDerivedDefinitions(baseClass)) {
      CudaTileEnumAttr enumAttr;
      StringRef className = record->getValueAsString("className");
      enumAttr.enumName = className.str();
      enumAttr.attrName = className.str() + "Attr";

      if (record->getValue("sinceVersion") &&
          !record->isValueUnset("sinceVersion")) {
        enumAttr.sinceVersion = record->getValueAsString("sinceVersion").str();
      }
      enumAttr.skipsVersionCheck = enumAttr.sinceVersion.empty();

      if (enumAttr.skipsVersionCheck) {
        structure.enumAttrs.emplace_back(std::move(enumAttr));
        continue;
      }

      // Collect enum cases with version info.
      for (const auto *caseInit : *record->getValueAsListInit("enumerants")) {
        const auto *caseRecord = cast<DefInit>(caseInit)->getDef();

        if (!caseRecord->getValue("sinceVersion") ||
            caseRecord->isValueUnset("sinceVersion") ||
            caseRecord->getValueAsString("sinceVersion").empty())
          PrintFatalError(caseRecord->getLoc(),
                          "Enum case '" + caseRecord->getName().str() +
                              "' is missing or has empty 'sinceVersion'");

        EnumCase enumCase;
        enumCase.name = caseRecord->getValueAsString("symbol").str();
        enumCase.value = caseRecord->getValueAsInt("value");
        enumCase.sinceVersion =
            caseRecord->getValueAsString("sinceVersion").str();
        enumAttr.cases.emplace_back(std::move(enumCase));
      }

      structure.enumAttrs.emplace_back(std::move(enumAttr));
    }
  };

  processEnumDefs("CudaTileI32EnumAttr");
  processEnumDefs("CudaTileI64EnumAttr");

  // Sort by tag value for deterministic enum generation.
  llvm::sort(structure.bytecodeAttrs, [](const auto &a, const auto &b) {
    return a.tagValue < b.tagValue;
  });

  return structure;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

LogicalResult mlir::tblgen::validateAttrTagAssignments(
    const RecordKeeper &records, const BytecodeAttrStructure &structure) {
  DenseSet<StringRef> attrsWithTags;
  for (const auto &attr : structure.bytecodeAttrs)
    attrsWithTags.insert(attr.attrName);

  for (const Record *record :
       records.getAllDerivedDefinitions("CudaTileAttrDef")) {
    StringRef tagName = extractAttrTagName(record->getName());
    if (!attrsWithTags.contains(tagName))
      PrintFatalError(record->getLoc(), "CudaTileAttrDef '" +
                                            record->getName().str() +
                                            "' is missing BytecodeAttrTag in "
                                            "BytecodeAttrOpcodes.td");
  }

  return success();
}
