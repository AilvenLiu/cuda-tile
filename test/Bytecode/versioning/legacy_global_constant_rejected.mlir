// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s 2>&1 | FileCheck %s

cuda_tile.module @kernels {
  global constant @g <f32: 1.0> : tile<1xf32>
}

// CHECK: error: constant global `g` cannot be encoded in bytecode version 13.1 (the `constant` attribute requires bytecode 13.3+)
