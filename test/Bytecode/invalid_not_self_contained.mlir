// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -split-input-file -verify-diagnostics -allow-unregistered-dialect %s

// expected-error @below{{only ops from the 'cuda_tile' dialect are allowed}}
cuda_tile.module @kernels {
  cuda_tile.entry @kernel() {
    // expected-note @below{{invalid op}}
    "test.op_from_different_dialect"() : () -> ()
  }
}

// -----

cuda_tile.module @kernels {
  // expected-error @below{{non-symbol operations are not allowed in a module body}}
  cuda_tile.constant <f32: 5.0> : !cuda_tile.tile<f32>
}
