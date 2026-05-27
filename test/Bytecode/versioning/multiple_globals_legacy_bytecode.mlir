// 13.1 / 13.2. The writer must emit four varints per global (legacy layout), not six.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -o %t.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.bc | FileCheck %s

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.2 %s -o %t2.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t2.bc | FileCheck %s

cuda_tile.module @kernels {
  global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
  global @val2 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
}

// CHECK: cuda_tile.module @kernels
// CHECK: global @val <
// CHECK: global @val2 <
