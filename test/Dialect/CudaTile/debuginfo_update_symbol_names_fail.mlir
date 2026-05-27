// RUN: cuda-tile-opt --test-debuginfo-update-symbol-name="symbol-to-rename=kernels::test_func scope=kernels::foo new-name=_fqZtest_func_f32 update-uses=true" --mlir-print-debuginfo --verify-diagnostics %s

// NOTE: We don't actually need any of the debuginfo stuff here, we're just checking a precondition that should already exist.

cuda_tile.module @kernels {
  // expected-error@below {{attempted to update symbol names in a scope that did not enclose the symbol itself}}
  entry @test_func() {
    return
  }

  // expected-note@below {{scope defined here}}
  entry @foo() {
    return
  }
} loc(unknown)
