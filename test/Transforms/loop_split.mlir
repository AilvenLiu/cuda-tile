// RUN: cuda-tile-opt %s --pass-pipeline='builtin.module(cuda_tile.module(cuda_tile.entry(loop-split)))'  --split-input-file | FileCheck %s

// LoopSplit is enabled for loop - unsupported due to comparison of non-iv with invariant
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @unsupported_cmp_non_iv
  cuda_tile.module @unsupported_cmp_non_iv {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %70, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - unsupported due to comparison of iv with non-invariant
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @unsupported_cmp_non_inv
  cuda_tile.module @unsupported_cmp_non_inv {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %70, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - sge predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_sge
  cuda_tile.module @split_sge {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - slt predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_slt
  cuda_tile.module @split_slt {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - sle predicate split
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_sle
  cuda_tile.module @split_sle {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - continue inside if-block
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_continue
  cuda_tile.module @split_continue {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT:    if
      // CHECK:        %[[MUL:.*]] = muli {{.*}}, {{.*}} : tile<i32>
      // CHECK-NEXT:   continue %[[MUL]] : tile<i32>
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than_or_equal %3, %arg1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          continue %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          continue %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop - CmpOp with uses
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_cmp_uses
  cuda_tile.module @split_cmp_uses {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:      {{.*}} = negi %[[FALSE]] : tile<i1>
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:      %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:      {{.*}} = negi %[[TRUE]] : tile<i1>
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %n = negi %5: tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop, IfOp requesting split is inside another IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_split_inner_if
  cuda_tile.module @supported_split_inner_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:      {{.*}} = if {{.*}} {
      // CHECK:        {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:      %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:      {{.*}} = if {{.*}} {
      // CHECK:        {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %70, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %100 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          %96 = if %100 -> (tile<i32>) {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            %99 = muli %c7, %920 : tile<i32>
            yield %99 : tile<i32>
          } else {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            yield %920 : tile<i32>
          }
          yield %96 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit is enabled for loop, splitting with IfOp inside IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_supported_nested_if
  cuda_tile.module @split_supported_nested_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        {{.*}} = if {{.*}}
      // CHECK-NOT:    {{.*}} = if {{.*}}
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          %96 = if %5 -> (tile<i32>) {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            %99 = muli %c7, %920 : tile<i32>
            yield %99 : tile<i32>
          } else {
            %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
            yield %920 : tile<i32>
          }
          yield %96 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// Loop split enabled - branch is inside inner loop
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_if_inside_nested_for
  cuda_tile.module @supported_if_inside_nested_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:        {{.*}} = for {{.*}}
      // CHECK:          {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:        %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:        {{.*}} = for {{.*}}
      // CHECK:          {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %99 = for %arg2 in (%1 to %0, step %2) : tile<i64> iter_values(%100 = %7) -> (tile<i32>) {
          %6 = if %5 -> (tile<i32>) {
            %9 = muli %c7, %c8 : tile<i32>
            yield %9 : tile<i32>
          } else {
            %96 = if %5 -> (tile<i32>) {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              %99 = muli %c7, %920 : tile<i32>
              yield %99 : tile<i32>
            } else {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              yield %920 : tile<i32>
            }
            yield %96 : tile<i32>
          }
          %80 = addi %6, %100 : tile<i32>
          continue %80 : tile <i32>
        }
        %8 = addi %7, %99 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// Check supported splitting of inner ForOp inside IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @supported_for_inside_if
  cuda_tile.module @supported_for_inside_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[ADD:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK-NEXT: %[[SPLITU:.*]] = mini %[[ADD]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[ADD]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      %[[SPLITIU:.*]] = mini %[[SPLITI:.*]], {{.*}} signed : tile<i64>
      // CHECK:      %[[SPLITIL:.*]] = maxi %[[SPLITI]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITIU]], step {{.*}})
      // CHECK: {{.*}} = for {{.*}} in (%[[SPLITIL]] to {{.*}}, step {{.*}})
      // CHECK: {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %77 = if %5 -> (tile<i32>) {
          yield %c8 : tile<i32>
        } else {
          %99 = for %arg2 in (%1 to %0, step %2) : tile<i64> iter_values(%100 = %7) -> (tile<i32>) {
            %11 = cmpi greater_than %arg2, %3, signed : tile<i64> -> tile<i1>
            %6 = if %11 -> (tile<i32>) {
              %9 = muli %c7, %c8 : tile<i32>
              yield %9 : tile<i32>
            } else {
              %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
              %99 = muli %c7, %920 : tile<i32>
              yield %99 : tile<i32>
            }
            %80 = addi %6, %100 : tile<i32>
            continue %80 : tile <i32>
          }
          yield %99 : tile<i32>
        }
        %8 = addi %7, %77 : tile<i32>
        continue %8 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_if
  cuda_tile.module @hint_disable_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        } {cuda_tile.loop_split = 0}
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        } {cuda_tile.loop_split = 0}
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for ForOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_for
  cuda_tile.module @hint_disable_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 0}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// LoopSplit disabled by hint for EntryOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_entry
  cuda_tile.module @hint_disable_entry {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%3 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } 
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----

// LoopSplit disabled by hint for EntryOp but enabled by hint for ForOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_entry_enable_for
  cuda_tile.module @hint_disable_entry_enable_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 1}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----

// LoopSplit disabled by hint for ForOp but enabled by hint for IfOp
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @hint_disable_for_enable_if
  cuda_tile.module @hint_disable_for_enable_if {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than_or_equal %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        } {cuda_tile.loop_split = 1}
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      } {cuda_tile.loop_split = 0}
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    } {cuda_tile.loop_split = 0}
  }
}

// -----
// LoopSplit is enabled for loop - unsigned comparison unsupported
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_unsigned
  cuda_tile.module @split_unsigned {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi less_than %arg1, %3, unsigned : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----
// LoopSplit is enabled for loop - split with non-1 step
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_step
  cuda_tile.module @split_step {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 4> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[ADDI:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK-NEXT: %[[SUBI:.*]] = subi %[[ADDI]], %[[LB:.*]] : tile<i64>
      // CHECK-NEXT: %[[DIVI:.*]] = divi %[[SUBI]], %[[STEP:.*]] signed rounding<positive_inf> : tile<i64>
      // CHECK-NEXT: %[[MULI:.*]] = muli %[[DIVI]], %[[STEP]] : tile<i64>
      // CHECK-NEXT: %[[SPLIT:.*]] = addi %[[LB]], %[[MULI]] : tile<i64>
      // CHECK-NEXT: %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], %[[LB]] signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_if_inside_while_loop
  cuda_tile.module @split_if_inside_while_loop {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLIT:.*]] = addi {{.*}}, {{.*}} : tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:        %[[FALSE:.*]] = constant <i1: false> : tile<i1>
      // CHECK:        {{.*}} = loop {{.*}}
      // CHECK:          {{.*}} = if %[[FALSE]] -> (tile<i32>) {
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK:        %[[TRUE:.*]] = constant <i1: true> : tile<i1>
      // CHECK:        {{.*}} = loop {{.*}}
      // CHECK:          {{.*}} = if %[[TRUE]] -> (tile<i32>) {
      %4 = for %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1) -> (tile<i32>) {
        %70 = addi %arg1, %arg1 : tile<i64>
        %5 = cmpi greater_than %arg1, %3, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %loop = loop iter_values(%arg2 = %c1) : tile<i32> -> tile<i32> {
          %6 = if %5 -> (tile<i32>) {
            %9 = muli %c7, %c8 : tile<i32>
            yield %9 : tile<i32>
          } else {
            yield %c7 : tile<i32>
          }
          break %6 : tile<i32>
        }
        %8 = addi %7, %loop : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

  // ****************** Test IV on Right Side with LESS_THAN ******************
  // Coverage: LoopSplit.cpp:53-54 (normalizeForOpCmp LESS_THAN predicate swap when IV on right)
  module attributes {gpu.container_module} {
    // CHECK: cuda_tile.module @split_iv_on_right_lt
    cuda_tile.module @split_iv_on_right_lt {
      entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
        %c0 = constant <i64: 0> : !cuda_tile.tile<i64>
        %c1 = constant <i64: 1> : !cuda_tile.tile<i64>
        %c10 = constant <i64: 10> : !cuda_tile.tile<i64>
        %c5 = constant <i64: 5> : !cuda_tile.tile<i64>
        %c_i32_1 = constant <i32: 1> : !cuda_tile.tile<i32>
        // less_than %c5, %iv normalizes to greater_than → split point = c5 + 1
        // CHECK:      addi
        // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed
        // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed
        // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
        // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
        // CHECK-NOT: cmpi
        %result = for %iv in (%c0 to %c10, step %c1) : tile<i64> iter_values(%init = %c_i32_1) -> (tile<i32>) {
          // IV on right side: %c5 < %iv normalizes to %iv > %c5
          %cond = cmpi less_than %c5, %iv, signed : tile<i64> -> tile<i1>
          %val = if %cond -> (tile<i32>) {
            %2 = constant <i32: 100> : !cuda_tile.tile<i32>
            yield %2 : tile<i32>
          } else {
            yield %init : tile<i32>
          }
          continue %val : tile<i32>
        }
        return
      }
    }
  }

  // -----

  // ****************** Test IV on Right Side with GREATER_THAN ******************
  // Coverage: LoopSplit.cpp:59-60 (normalizeForOpCmp GREATER_THAN predicate swap when IV on right)
  module attributes {gpu.container_module} {
    // CHECK: cuda_tile.module @split_iv_on_right_gt
    cuda_tile.module @split_iv_on_right_gt {
      entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
        %c0 = constant <i64: 0> : !cuda_tile.tile<i64>
        %c1 = constant <i64: 1> : !cuda_tile.tile<i64>
        %c10 = constant <i64: 10> : !cuda_tile.tile<i64>
        %c5 = constant <i64: 5> : !cuda_tile.tile<i64>
        %c_i32_1 = constant <i32: 1> : !cuda_tile.tile<i32>
        // greater_than %c5, %iv normalizes to less_than → split point = c5
        // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed
        // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed
        // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
        // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
        // CHECK-NOT: cmpi
        %result = for %iv in (%c0 to %c10, step %c1) : tile<i64> iter_values(%init = %c_i32_1) -> (tile<i32>) {
          // IV on right side: %c5 > %iv normalizes to %iv < %c5
          %cond = cmpi greater_than %c5, %iv, signed : tile<i64> -> tile<i1>
          %val = if %cond -> (tile<i32>) {
            %2 = constant <i32: 100> : !cuda_tile.tile<i32>
            yield %2 : tile<i32>
          } else {
            yield %init : tile<i32>
          }
          continue %val : tile<i32>
        }
        return
      }
    }
  }

  // -----

  // ****************** Test IV on Right Side with GREATER_THAN_OR_EQUAL ******************
  // Coverage: LoopSplit.cpp:62-63 (normalizeForOpCmp GREATER_THAN_OR_EQUAL predicate swap when IV on right)
  module attributes {gpu.container_module} {
    // CHECK: cuda_tile.module @split_iv_on_right_ge
    cuda_tile.module @split_iv_on_right_ge {
      entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
        %c0 = constant <i64: 0> : !cuda_tile.tile<i64>
        %c1 = constant <i64: 1> : !cuda_tile.tile<i64>
        %c10 = constant <i64: 10> : !cuda_tile.tile<i64>
        %c5 = constant <i64: 5> : !cuda_tile.tile<i64>
        %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
        %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
        %c_i32_1 = constant <i32: 1> : !cuda_tile.tile<i32>
        // greater_than_or_equal %c5, %iv normalizes to less_than_or_equal → split point = c5 + 1
        // CHECK:      addi
        // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed
        // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed
        // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
        // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
        // CHECK-NOT: cmpi
        %result = for %iv in (%c0 to %c10, step %c1) : tile<i64> iter_values(%init = %c_i32_1) -> (tile<i32>) {
          // IV on right side: %c5 >= %iv normalizes to %iv <= %c5
          %cond = cmpi greater_than_or_equal %c5, %iv, signed : tile<i64> -> tile<i1>
          %ptr_int = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
          %addr = addi %ptr_int, %iv : tile<i64>
          %ptr = int_to_ptr %addr : tile<i64> -> tile<ptr<i32>>
          %val = if %cond -> (tile<i32>) {
            %mul = muli %c7, %c8 : tile<i32>
            yield %mul : tile<i32>
          } else {
            yield %c7 : tile<i32>
          }
          %sum = addi %init, %val : tile<i32>
          continue %sum : tile<i32>
        }
        return
      }
    }
  }

// -----

// Unsigned ForOp with matching unsigned cmpi (should split)
// Coverage: LoopSplit.cpp:38 b1 (unsigned ternary), :272 b1 (unsigned in performLoopSplit)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_unsigned_for
  cuda_tile.module @split_unsigned_for {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c1_i32 = constant <i32: 0> : !cuda_tile.tile<i32>
      %0 = constant <i64: 128> : !cuda_tile.tile<i64>
      %1 = constant <i64: 0> : !cuda_tile.tile<i64>
      %2 = constant <i64: 1> : !cuda_tile.tile<i64>
      %3 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} unsigned : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} unsigned : tile<i64>
      // CHECK-NEXT: {{.*}} = for unsigned {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for unsigned {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for unsigned %arg1 in (%1 to %0, step %2) : tile<i64> iter_values(%7 = %c1_i32) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %3, unsigned : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// IfOp condition is a constant (not a CmpIOp) — should not split
// Coverage: LoopSplit.cpp:117 b1 (!cmp path)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @no_split_non_cmpi_condition
  cuda_tile.module @no_split_non_cmpi_condition {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %cond_const = constant <i1: true> : !cuda_tile.tile<i1>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %6 = if %cond_const -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// cmpi equal with IV on LHS — unsupported predicate, should not split
// Coverage: LoopSplit.cpp:135 b2 (default case in normalizedPred switch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @no_split_equal_predicate
  cuda_tile.module @no_split_equal_predicate {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi equal %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// cmpi equal with IV on RHS — unsupported predicate in normalizeForOpCmp switch default
// Coverage: LoopSplit.cpp:52 b4 (default case in reverse-predicate switch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @no_split_equal_iv_on_rhs
  cuda_tile.module @no_split_equal_iv_on_rhs {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi equal %c32, %arg1, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// cmpi with block argument as RHS (rhs has no defining op) — should split
// Coverage: LoopSplit.cpp:130 b1 (rhsOp is null, short-circuit in && check)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_iv_cmp_block_arg
  cuda_tile.module @split_iv_cmp_block_arg {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>, %arg1: !cuda_tile.tile<i64>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %iv in (%lb to %ub, step %step) : tile<i64> iter_values(%acc = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %iv, %arg1, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %iv : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %9 = muli %c7, %c8 : tile<i32>
          yield %9 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %acc, %6 : tile<i32>
        %96 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          %99 = muli %c7, %920 : tile<i32>
          yield %99 : tile<i32>
        } else {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        }
        %98 = addi %8, %96 : tile<i32>
        continue %98 : tile<i32>
      }
      %10 = addi %4, %c7 : tile<i32>
      %20 = store_ptr_tko weak %arg0, %10 : tile<ptr<i32>>, tile<i32> -> token
      return
    }
  }
}

// -----

// Split not profitable: high threshold hint + small if body, no heavy ops
// Coverage: LoopSplit.cpp:172 b1 (!isProfitable), also exercises countOps lambda
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @no_split_unprofitable
  cuda_tile.module @no_split_unprofitable {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK: {{.*}} = for {{.*}} in ({{.*}} to {{.*}}, step {{.*}})
      // CHECK-NOT: {{.*}} = for {{.*}}
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          yield %c7 : tile<i32>
        } else {
          yield %7 : tile<i32>
        } {cuda_tile.loop_split = 10}
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split profitable: threshold > 1, then block has enough ops (thenSize >= threshold)
// Coverage: LoopSplit.cpp:101 (thenSize >= threshold true branch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_profitable_large_then
  cuda_tile.module @split_profitable_large_then {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          %a = addi %c7, %c8 : tile<i32>
          %b = muli %a, %c7 : tile<i32>
          %c = addi %b, %c8 : tile<i32>
          yield %c : tile<i32>
        } else {
          yield %c7 : tile<i32>
        } {cuda_tile.loop_split = 3}
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split profitable: threshold > 1, if body has heavy op (hasHeavyOps = true)
// Coverage: LoopSplit.cpp:101 (hasHeavyOps true branch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_profitable_heavy_ops
  cuda_tile.module @split_profitable_heavy_ops {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK-NOT:  addi
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %40 = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
        %30 = addi %40, %arg1 : tile<i64>
        %50 = int_to_ptr %30 : tile<i64> -> tile<ptr<i32>>
        %6 = if %5 -> (tile<i32>) {
          %920:2 = load_ptr_tko weak %50 : tile<ptr<i32>> -> tile<i32>, token
          yield %920 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        } {cuda_tile.loop_split = 10}
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split profitable: threshold > 1, else block has enough ops (elseSize >= threshold)
// Coverage: LoopSplit.cpp:101 (elseSize >= threshold true branch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_profitable_large_else
  cuda_tile.module @split_profitable_large_else {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c8 = constant <i32: 8> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 1> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // CHECK:      %[[SPLITU:.*]] = mini %[[SPLIT:.*]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi %[[SPLIT]], {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          yield %c7 : tile<i32>
        } else {
          %a = addi %c7, %c8 : tile<i32>
          %b = muli %a, %c7 : tile<i32>
          %c = addi %b, %c8 : tile<i32>
          yield %c : tile<i32>
        } {cuda_tile.loop_split = 3}
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split with step != 1: triggers step alignment logic and isConstOne false path
// Coverage: LoopSplit.cpp:280 (isConstOne returns false), :315 (step alignment branch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_step_not_one
  cuda_tile.module @split_step_not_one {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %step = constant <i64: 2> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      // Step alignment: splitPoint = lb + Ceil(splitPoint - lb, step) * step
      // CHECK:      subi
      // CHECK:      divi
      // CHECK:      muli
      // CHECK:      addi
      // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          yield %c7 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split with dynamic (non-constant) step: step from ptr_to_int, not ConstantOp
// Coverage: LoopSplit.cpp:315 b0 (!constStep true branch)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @split_dynamic_step
  cuda_tile.module @split_dynamic_step {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c7 = constant <i32: 7> : !cuda_tile.tile<i32>
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %ub = constant <i64: 128> : !cuda_tile.tile<i64>
      %lb = constant <i64: 0> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      %step = ptr_to_int %arg0 : tile<ptr<i32>> -> tile<i64>
      // Step alignment: splitPoint = lb + Ceil(splitPoint - lb, step) * step
      // CHECK:      subi
      // CHECK:      divi
      // CHECK:      muli
      // CHECK:      addi
      // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed : tile<i64>
      // CHECK-NEXT: {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %4 = for %arg1 in (%lb to %ub, step %step) : tile<i64> iter_values(%7 = %c0) -> (tile<i32>) {
        %5 = cmpi less_than %arg1, %c32, signed : tile<i64> -> tile<i1>
        %6 = if %5 -> (tile<i32>) {
          yield %c7 : tile<i32>
        } else {
          yield %c7 : tile<i32>
        }
        %8 = addi %7, %6 : tile<i32>
        continue %8 : tile<i32>
      }
      return
    }
  }
}

// -----

// Split with loop-carried step (BlockArgument from iter_values)
// Coverage: LoopSplit.cpp:315 b0 (!constStep true branch via non-ConstantOp step)
module attributes {gpu.container_module} {
  // CHECK: cuda_tile.module @crash_step_from_iter_arg
  cuda_tile.module @crash_step_from_iter_arg {
    entry @kernel_0(%arg0: !cuda_tile.tile<ptr<i32>>) {
      %c0 = constant <i32: 0> : !cuda_tile.tile<i32>
      %c0_i64 = constant <i64: 0> : !cuda_tile.tile<i64>
      %c1 = constant <i64: 1> : !cuda_tile.tile<i64>
      %c2 = constant <i64: 2> : !cuda_tile.tile<i64>
      %c64 = constant <i64: 64> : !cuda_tile.tile<i64>
      %c32 = constant <i64: 32> : !cuda_tile.tile<i64>
      %c128 = constant <i64: 128> : !cuda_tile.tile<i64>
      // Dynamic step alignment with loop-carried %dyn_step
      // CHECK:      subi
      // CHECK:      divi {{.*}} rounding<positive_inf>
      // CHECK:      muli
      // CHECK:      addi
      // CHECK:      %[[SPLITU:.*]] = mini {{.*}} signed : tile<i64>
      // CHECK-NEXT: %[[SPLITL:.*]] = maxi {{.*}} signed : tile<i64>
      // CHECK:      {{.*}} = for {{.*}} in ({{.*}} to %[[SPLITU]], step {{.*}})
      // CHECK:      {{.*}} = for {{.*}} in (%[[SPLITL]] to {{.*}}, step {{.*}})
      // CHECK-NOT: if
      %outer:2 = for %outer_iv in (%c0_i64 to %c64, step %c1) : tile<i64>
          iter_values(%acc = %c0, %dyn_step = %c2) -> (tile<i32>, tile<i64>) {
        %inner = for %inner_iv in (%c0_i64 to %c128, step %dyn_step) : tile<i64> iter_values(%iacc = %c0) -> (tile<i32>) {
          %cond = cmpi less_than %inner_iv, %c32, signed : tile<i64> -> tile<i1>
          %val = if %cond -> (tile<i32>) {
            %x = constant <i32: 42> : !cuda_tile.tile<i32>
            yield %x : tile<i32>
          } else {
            yield %iacc : tile<i32>
          }
          continue %val : tile<i32>
        }
        %sum = addi %acc, %inner : tile<i32>
        continue %sum, %dyn_step : tile<i32>, tile<i64>
      }
      return
    }
  }
}
