// RUN: not cuda-tile-translate -mlir-to-cudatilebc %s -no-implicit-module 2>&1 | FileCheck %s

// Verify that programs using non-TileIR types in control-flow terminators
// (yield, break, continue) are rejected by the bytecode writer. The MLIR
// verifier accepts AnyType in these ops (to support foreign frontends that
// embed non-CudaTile ops), but the bytecode format is a closed format that
// only supports CudaTile and builtin scalar/float types — so serialization
// must fail rather than silently produce an unreadable file.

// CHECK: unsupported type in bytecode writer

cuda_tile.module @kernels {

  // yield: foreign type flows from a function arg through an if-result.
  cuda_tile.entry @yield_foreign_type(%val: index, %cond: !cuda_tile.tile<i1>) {
    %r = cuda_tile.if %cond -> (index) {
      cuda_tile.yield %val : index
    } else {
      cuda_tile.yield %val : index
    }
    cuda_tile.return
  }

  // break: foreign type flows from a function arg out of a loop via break.
  cuda_tile.entry @break_foreign_type(%val: index) {
    %r = cuda_tile.loop : index {
      cuda_tile.break %val : index
    }
    cuda_tile.return
  }

  // continue: foreign type flows from a function arg as a loop-carried value.
  cuda_tile.entry @continue_foreign_type(%val: index) {
    cuda_tile.loop iter_values(%arg = %val) : index {
      cuda_tile.continue %arg : index
    }
    cuda_tile.return
  }

}
