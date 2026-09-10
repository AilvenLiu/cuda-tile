// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {

  //===--------------------------------------------------------------------===//
  // load_view_tko
  //===--------------------------------------------------------------------===//

cuda_tile.entry @load_view_in_bound_2d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
        : tensor_view<8192x128xf32, strides=[128, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    %a, %atok = load_view_tko weak %view[%c0, %c0] inbounds = [true, false]
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> tile<64x64xf32>, token
    %b, %btok = load_view_tko weak %view[%c0, %c0] inbounds = [false, true]
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> tile<64x64xf32>, token
    %c, %ctok = load_view_tko weak %view[%c0, %c0] inbounds = [true, true]
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> tile<64x64xf32>, token
}

cuda_tile.entry @load_view_in_bound_3d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    %t, %tok = load_view_tko weak %view[%c0, %c0, %c0] inbounds = [true, false, true]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token
}

// `inbounds` is required in the IR. When the textual form omits the
// `inbounds = [...]` clause, the parser synthesizes an all-`false` vector
// of the correct rank, and the printer elides the clause for the same
// all-`false` case so that omission round-trips through cuda-tile-opt.
// The bytecode writer, however, always encodes the attribute.
cuda_tile.entry @load_view_in_bound_no_attribute(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    %t, %tok = load_view_tko weak %view[%c0, %c0, %c0]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token
}

// Explicit all-`false` form is elided by the printer (same as the
// previous case) and must still round-trip without losing information in
// the bytecode.
cuda_tile.entry @load_view_in_bound_all_false(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv2d = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
        : tensor_view<8192x128xf32, strides=[128, 1]>
    %view2d = make_partition_view %tv2d
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>
    %tv3d = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view3d = make_partition_view %tv3d
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>

    %a, %atok = load_view_tko weak %view2d[%c0, %c0] inbounds = [false, false]
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> tile<64x64xf32>, token
    %b, %btok = load_view_tko weak %view3d[%c0, %c0, %c0] inbounds = [false, false, false]
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> tile<8x8x8xf32>, token
}

  //===--------------------------------------------------------------------===//
  // store_view_tko
  //===--------------------------------------------------------------------===//

cuda_tile.entry @store_view_in_bound_2d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
        : tensor_view<8192x128xf32, strides=[128, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %val = constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>

    %t0 = store_view_tko weak %val, %view[%c0, %c0] inbounds = [true, false]
        : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> token
    %t1 = store_view_tko weak %val, %view[%c0, %c0] inbounds = [false, true]
        : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> token
    %t2 = store_view_tko weak %val, %view[%c0, %c0] inbounds = [true, true]
        : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> token
}

cuda_tile.entry @store_view_in_bound_3d(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %val = constant <f32: 0.0> : !cuda_tile.tile<8x8x8xf32>

    %t = store_view_tko weak %val, %view[%c0, %c0, %c0] inbounds = [true, false, true]
        : tile<8x8x8xf32>, partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> token
}

// `inbounds` is required in the IR. When the textual form omits the
// `inbounds = [...]` clause, the parser synthesizes an all-`false` vector
// of the correct rank, and the printer elides the clause for the same
// all-`false` case so that omission round-trips through cuda-tile-opt.
// The bytecode writer, however, always encodes the attribute.
cuda_tile.entry @store_view_in_bound_3d_no_attribute(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view = make_partition_view %tv
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %val = constant <f32: 0.0> : !cuda_tile.tile<8x8x8xf32>

    %t = store_view_tko weak %val, %view[%c0, %c0, %c0]
        : tile<8x8x8xf32>, partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> token
}

// Explicit all-`false` form is elided by the printer (same as the
// previous case) and must still round-trip without losing information in
// the bytecode.
cuda_tile.entry @store_view_in_bound_all_false(%ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv2d = make_tensor_view %ptr, shape=[8192, 128], strides=[128, 1]
        : tensor_view<8192x128xf32, strides=[128, 1]>
    %view2d = make_partition_view %tv2d
        : partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>
    %tv3d = make_tensor_view %ptr, shape=[256, 128, 64], strides=[8192, 64, 1]
        : tensor_view<256x128x64xf32, strides=[8192, 64, 1]>
    %view3d = make_partition_view %tv3d
        : partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>
    %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
    %val2d = constant <f32: 0.0> : !cuda_tile.tile<64x64xf32>
    %val3d = constant <f32: 0.0> : !cuda_tile.tile<8x8x8xf32>

    %t0 = store_view_tko weak %val2d, %view2d[%c0, %c0] inbounds = [false, false]
        : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<8192x128xf32, strides=[128, 1]>>, tile<i32>
        -> token
    %t1 = store_view_tko weak %val3d, %view3d[%c0, %c0, %c0] inbounds = [false, false, false]
        : tile<8x8x8xf32>, partition_view<tile=(8x8x8), tensor_view<256x128x64xf32, strides=[8192, 64, 1]>>, tile<i32>
        -> token
}

}
