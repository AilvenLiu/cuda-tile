//===- BytecodeVerification.cpp - Bytecode format verification //-*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cuda_tile/Bytecode/Common/BytecodeVerification.h"

#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

using namespace mlir;
using namespace mlir::cuda_tile;

LogicalResult
cuda_tile::verifySelfContainedModuleAndOperationInvariants(ModuleOp module) {
  if (module.getBody().empty()) {
    return module.emitOpError("module body is empty");
  }

  // Validate that we have a self-contained module that matches what we can
  // encode within the bytecode (e.g. no non-functions/globals nested in the
  // module).
  for (Operation &op : module.getBody().front()) {
    if (!isa<FunctionOpInterface, GlobalOp>(&op)) {
      auto diag = module.emitOpError(
          "only function and global ops are allowed in the body");
      diag.attachNote(op.getLoc()) << "invalid op: " << op.getName();
      return diag;
    }
  }

  // Allow only ops from the CudaTile dialect inside of the module (at any
  // nesting level).
  Dialect *dialect = module->getDialect();
  WalkResult status = module->walk([&](Operation *op) {
    if (op->getDialect() != dialect) {
      auto diag = module.emitOpError("only ops from the '")
                  << dialect->getNamespace() << "' dialect are allowed";
      diag.attachNote(op->getLoc()) << "invalid op: " << op->getName();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return status.wasInterrupted() ? failure() : success();
}
