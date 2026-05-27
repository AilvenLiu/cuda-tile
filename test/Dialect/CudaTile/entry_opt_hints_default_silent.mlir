// Test that invalid/unsupported hints are silently ignored by default (no flags)
// RUN: cuda-tile-opt %s | FileCheck %s

// CHECK-LABEL: cuda_tile.module @test_default_silent
cuda_tile.module @test_default_silent {
  // CHECK: entry @unknown_sm_key(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100a = {num_cta_in_cga = 2}>
  entry @unknown_sm_key(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100a={num_cta_in_cga=2}> {
    return
  }

  // CHECK: entry @unknown_param(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_qqq = 1}>
  entry @unknown_param(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_qqq=1}> {
    return
  }

  // CHECK: entry @invalid_value(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 7}>
  entry @invalid_value(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=7}> {
    return
  }

  // CHECK: entry @invalid_type(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = "a"}>
  entry @invalid_type(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga="a"}> {
    return
  }

  // CHECK: entry @invalid_warps_value(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 2, simt_num_cta_in_cga = 9}>
  entry @invalid_warps_value(%arg0: !cuda_tile.tile<ptr<f32>>) optimization_hints=<sm_100={num_cta_in_cga=2, simt_num_cta_in_cga=9}> {
    return
  }
}

