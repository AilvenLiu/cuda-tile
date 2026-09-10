// RUN: cuda-tile-opt %s -verify-diagnostics -split-input-file

// MakeTensorViewOp requires the base pointer and the result tensor_view to
// agree on whether a ptr_attr is present. `ptr_attr<none>` is a distinct
// spelling from carrying no ptr_attr, so these cases are reachable with
// `none` alone.

//===----------------------------------------------------------------------===//
// MakeTensorViewOp — ptr_attr<none> base must preserve the attribute
//===----------------------------------------------------------------------===//

cuda_tile.module @base_none_result_absent_module {
  cuda_tile.entry @base_none_result_absent(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32, #cuda_tile.ptr_attr<none>>>) {
    // expected-error @below{{tensor_view must preserve ptr_attr from base pointer}}
    %tv = cuda_tile.make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tile<ptr<f32, #cuda_tile.ptr_attr<none>>> ->
          tensor_view<32x32xf32, strides=[32,1]>
    cuda_tile.return
  }
}

// -----

//===----------------------------------------------------------------------===//
// MakeTensorViewOp — base without ptr_attr cannot produce ptr_attr<none>
//===----------------------------------------------------------------------===//

cuda_tile.module @base_absent_result_none_module {
  cuda_tile.entry @base_absent_result_none(
      %base: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    // expected-error @below{{pointer base without ptr_attr cannot produce tensor_view with ptr_attr}}
    %tv = cuda_tile.make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tensor_view<32x32xf32, strides=[32,1], #cuda_tile.ptr_attr<none>>
    cuda_tile.return
  }
}
