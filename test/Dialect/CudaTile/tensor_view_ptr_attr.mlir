// RUN: %round_trip_test %s %t

cuda_tile.module @kernels {
  // TensorViewType without PtrAttr
  entry @tv_no_attr(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tile<ptr<f32>> ->
          tensor_view<32x32xf32, strides=[32,1]>
    cuda_tile.return
  }

  // Regular tensor_view still works (backward compat)
  entry @tv_regular(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %tv = cuda_tile.make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tensor_view<32x32xf32, strides=[32,1]>
    cuda_tile.return
  }
}
