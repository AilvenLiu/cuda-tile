// Backward-compat regression for `inbounds` on `load_view_tko` and
// `store_view_tko` (added in 13.4).
//
// Round-trip property:
//
//   - The op printer elides the `inbounds = [...]` clause when every
//     element is `false` (`CudaTile.cpp::printViewTkoCommon`).
//   - The parser synthesizes the same all-`false` vector for an omitted
//     clause (`CudaTile.cpp::parseLoadViewTko/StoreViewTko`).
//   - The bytecode reader's per-operand-rank fixup synthesizes the same
//     all-`false` vector when reading legacy (< 13.4) bytecode
//     (`BytecodeReaderGen.cpp::generateSameOperandRankFixups`).
//   - The bytecode writer omits the attribute on the wire when the
//     runtime value is itself all-`false` and the target version is
//     pre-13.4 (`BytecodeGen.cpp::generateAttributeSerialization`,
//     `sameOperandRank` branch).
//
// Together these guarantee that an op whose `inbounds` is all-`false`
// round-trips losslessly through 13.1 bytecode: the writer omits, the
// reader resynthesizes, and the resulting in-memory attribute is
// identical to the one we started with.
//
// We print the post-round-trip IR in generic form to bypass the custom
// printer (which would also elide the all-`false` clause and hide the
// materialization).

// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.1 %s -o %t.bc
// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %t.bc \
// RUN:   | cuda-tile-opt -no-implicit-module -mlir-print-op-generic \
// RUN:   | FileCheck %s

// CHECK-LABEL: kernels
cuda_tile.module @kernels {

  // 3D index: the parser synthesizes inbounds = [false, false, false],
  // the writer omits it on the wire (target 13.1 < 13.4), and the
  // reader's fixup re-materializes a 3-element all-`false` vector sized
  // to the index rank.

  // CHECK-LABEL: view_no_inbounds_3d
  cuda_tile.entry @view_no_inbounds_3d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    // CHECK: "cuda_tile.load_view_tko"
    // CHECK-SAME: inbounds = array<i1: false, false, false>
    %t, %tok = load_view_tko weak %view[%c0, %c0, %c0]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token

    // CHECK: "cuda_tile.store_view_tko"
    // CHECK-SAME: inbounds = array<i1: false, false, false>
    %s = store_view_tko weak %t, %view[%c0, %c0, %c0]
        : tile<8x8x8xf32>, partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> token
  }

  // CHECK-LABEL: view_no_inbounds_1d
  cuda_tile.entry @view_no_inbounds_1d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[128], strides=[1]
        : tensor_view<128xf32, strides=[1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    // CHECK: "cuda_tile.load_view_tko"
    // CHECK-SAME: inbounds = array<i1: false>
    %t, %tok = load_view_tko weak %view[%c0]
        : partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32>
        -> tile<8xf32>, token

    // CHECK: "cuda_tile.store_view_tko"
    // CHECK-SAME: inbounds = array<i1: false>
    %s = store_view_tko weak %t, %view[%c0]
        : tile<8xf32>, partition_view<tile=(8), tensor_view<128xf32, strides=[1]>>, tile<i32>
        -> token
  }

  // CHECK-LABEL: view_explicit_all_false
  cuda_tile.entry @view_explicit_all_false(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    // CHECK: "cuda_tile.load_view_tko"
    // CHECK-SAME: inbounds = array<i1: false, false, false>
    %t, %tok = load_view_tko weak %view[%c0, %c0, %c0] inbounds = [false, false, false]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token

    // CHECK: "cuda_tile.store_view_tko"
    // CHECK-SAME: inbounds = array<i1: false, false, false>
    %s = store_view_tko weak %t, %view[%c0, %c0, %c0] inbounds = [false, false, false]
        : tile<8x8x8xf32>, partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> token
  }
}
