//===- TestPasses.cpp - Passes used for the unit tests ----------*- C++ -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "cuda_tile/Dialect/CudaTile/Transforms/TransformsUtils.h"

using namespace mlir;
using namespace cuda_tile;

namespace {
/// Provide a pass that updates the given symbol. The pass takes an option
struct TestDebugInfoUpdateSymbolName
    : public PassWrapper<TestDebugInfoUpdateSymbolName,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDebugInfoUpdateSymbolName);

  using Base =
      PassWrapper<TestDebugInfoUpdateSymbolName, OperationPass<ModuleOp>>;

  TestDebugInfoUpdateSymbolName() : Base() {}
  TestDebugInfoUpdateSymbolName(const TestDebugInfoUpdateSymbolName &other)
      : Base(other) {
    symbolToRename = other.symbolToRename;
    scope = other.scope;
    newName = other.newName;
    updateUses = other.updateUses;
  }

  StringRef getArgument() const final {
    return "test-debuginfo-update-symbol-name";
  }
  StringRef getDescription() const final {
    return "Test the debuginfo utilities to update symbol names";
  }

  /// Parse a string that looks like `outer::middle::inner` and produce a
  /// SymbolRefAttr from it.
  SymbolRefAttr parseNestedSpecifier(MLIRContext &ctx, StringRef ref) {
    llvm::SmallVector<FlatSymbolRefAttr> nestedSymbols;
    while (!ref.empty()) {
      auto front = ref.take_until([](char c) { return c == ':'; });
      ref = ref.drop_front(front.size());
      // The last symbol won't have the `::` but that's OK.
      (void)ref.consume_front("::");
      nestedSymbols.push_back(FlatSymbolRefAttr::get(&ctx, front));
    }

    return SymbolRefAttr::get(
        &ctx, nestedSymbols.front().getValue(),
        ArrayRef<FlatSymbolRefAttr>{nestedSymbols}.drop_front());
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    ModuleOp theModule = getOperation();
    SymbolTableCollection symtabs;

    // Construct the symbol ref attr for the thing to be renamed.
    auto toRenameAttr = parseNestedSpecifier(ctx, symbolToRename);

    // Same for the scope that we're replacing in.
    mlir::Operation *scopeOp = theModule;
    if (!scope.empty()) {
      auto scopeRef = parseNestedSpecifier(ctx, scope);
      scopeOp = symtabs.lookupSymbolIn(theModule, scopeRef);
      if (!scopeOp) {
        theModule->emitError()
            << "could not find symbol op with name '" << scopeRef << "'";
        return signalPassFailure();
      }
    }

    auto symOp =
        symtabs.lookupSymbolIn<SymbolOpInterface>(theModule, toRenameAttr);
    if (!symOp) {
      theModule->emitError()
          << "could not find symbol op with name '" << toRenameAttr << "'";
      return signalPassFailure();
    }

    // Pull out the symbol table we can use for renaming.
    auto symtab = symtabs.getSymbolTable(
        symOp->getParentWithTrait<OpTrait::SymbolTable>());

    // Try to do the rename.
    if (failed(updateSymbolName(StringAttr::get(&ctx, newName), symOp, scopeOp,
                                (updateUses ? &symtab : nullptr))))
      return signalPassFailure();
  }

private:
  Pass::Option<std::string> symbolToRename{
      *this, "symbol-to-rename",
      llvm::cl::desc(
          "The symbol op to rename. Accepts a nested symbol specifier "
          "like outer::middle::inner.")};

  Pass::Option<std::string> scope{
      *this, "scope",
      llvm::cl::desc(
          "The scope for the rename. Accepts a nested symbol specifier "
          "like outer::middle::inner.")};
  Pass::Option<std::string> newName{
      *this, "new-name",
      llvm::cl::desc(
          "New name for the symbol. For outer::middle::inner, this would "
          "only replace inner.")};
  Pass::Option<bool> updateUses{
      *this, "update-uses",
      llvm::cl::desc("If the pass should also attempt to update uses of the "
                     "symbol being renamed."),
      llvm::cl::init(false)};
};
} // namespace

namespace mlir::cuda_tile::test {
void registerTransformsUtilsTestPasses() {
  PassRegistration<TestDebugInfoUpdateSymbolName>{};
}
} // namespace mlir::cuda_tile::test
