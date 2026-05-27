// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.2 %s 2>&1 | FileCheck %s

cuda_tile.module @kernels {
  global private @dce_private_used <f32: 2.560000e+02> : tile<1xf32>
}

// CHECK: error: global `dce_private_used` uses non-public symbol visibility, which cannot be encoded in bytecode version 13.2 (requires bytecode 13.3+)
