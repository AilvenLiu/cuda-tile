// Test that encoding (mlir-to-cudatilebc) fails with -Wunsupported-hints -Werr-hints
// when the module has invalid/unsupported optimization hints (same behavior as decode).
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -Wunsupported-hints -Werr-hints -verify-diagnostics %s

cuda_tile.module @kernels {
  // expected-warning @below{{num_warps_in_cta is not known hint for current Operation}}
  // expected-error @below{{Optimization hints verification failed}}
  entry @test_optimization_hints(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_warps_in_cta = "a"}> {
    return
  }
}
