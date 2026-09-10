// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.entry(loop-split)))' | FileCheck %s

// Regression test: loop-split must not crash when IfOp has no else block.
// Previously, copyLoop() called region.front() on the empty else region of a
// then-only IfOp, causing a segfault. The split should produce two loops:
// the first inlines the then-block (condition always true), the second is
// empty (no else body to copy).
module attributes {gpu.container_module} {
  cuda_tile.module @split_if_no_else {
    // CHECK-LABEL: entry @kernel_0
    // CHECK-SAME:  (%[[ARG0:.+]]: {{.+}})
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %cst_7_i32 = constant <i32: 7> : !cuda_tile.tile<i32>
      %cst_0_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
      %cst_128_i64 = constant <i64: 128> : !cuda_tile.tile<i64>
      %cst_0_i64 = constant <i64: 0> : !cuda_tile.tile<i64>
      %cst_1_i64 = constant <i64: 1> : !cuda_tile.tile<i64>
      %cst_32_i64 = constant <i64: 32> : !cuda_tile.tile<i64>
      %for = for %loopIdx in (%cst_0_i64 to %cst_128_i64, step %cst_1_i64) : tile<i64> iter_values(%iterArg0 = %cst_0_i32) -> (tile<i32>) {
        %1 = cmpi less_than %loopIdx, %cst_32_i64, signed : tile<i64> -> tile<i1>
        if %1 {
          %2 = store_ptr_tko weak %arg0, %cst_7_i32 : !cuda_tile.tile<ptr<i32>>, !cuda_tile.tile<i32> -> token
        }
        continue %iterArg0 : tile<i32>
      }
      %0 = store_ptr_tko weak %arg0, %for : !cuda_tile.tile<ptr<i32>>, !cuda_tile.tile<i32> -> token
      return
    }
  }
}
// Two loops are produced; no `if` remains in the output.
// First loop runs [lb, min(split,ub)) and inlines the then-block (store always executes).
// Second loop runs [max(split,lb), ub) and has no body from the missing else-block.
// CHECK-DAG: %[[MIN:.+]] = mini {{.+}} signed
// CHECK-DAG: %[[MAX:.+]] = maxi {{.+}} signed
// CHECK:     %[[FOR1:.+]] = for %{{.+}} in (%{{.+}} to %[[MIN]],
// CHECK:       store_ptr_tko
// CHECK:       continue
// CHECK:     %[[FOR2:.+]] = for %{{.+}} in (%[[MAX]] to
// CHECK-NOT: if
// CHECK:       continue
