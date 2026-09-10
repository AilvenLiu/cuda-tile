// The writer is allowed to downgrade `load_view_tko` / `store_view_tko` to
// a pre-13.4 target ONLY when the runtime `inbounds` is all-`false`. Any
// element that is `true` carries information the legacy reader cannot
// reconstruct, so the writer must refuse with a diagnostic.

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:     -bytecode-version=13.1 -verify-diagnostics %s

cuda_tile.module @kernels {
  cuda_tile.entry @load_with_true_inbounds(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    // A single `true` element is sufficient to make the round-trip lossy.
    // expected-error@below {{operation requires bytecode version 13.4+ because it carries a non-default 'inbounds' attribute}}
    %t, %tok = load_view_tko weak %view[%c0, %c0, %c0] inbounds = [true, false, false]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token
  }
}
