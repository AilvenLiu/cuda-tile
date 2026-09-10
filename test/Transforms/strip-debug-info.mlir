// RUN: cuda-tile-opt %s --strip-debuginfo | FileCheck %s

// Verify that the strip-debug-info pass removes file/line location information
// from ops, functions, and modules, replacing them with loc(unknown).
// Without --mlir-print-debuginfo, loc(unknown) is not printed, so no loc()
// annotations should appear in the output.

// CHECK-NOT: loc("/tmp/foo.py"
// CHECK-NOT: #cuda_tile.di_loc

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">
#compile_unit = #cuda_tile.di_compile_unit<file = #file>
#subprogram = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_func",
  linkageName = "test_func",
  compileUnit = #compile_unit,
  scopeLine = 2
>
#di_loc = #cuda_tile.di_loc<loc("/tmp/foo.py":3:4) in #subprogram>

// CHECK-LABEL: cuda_tile.module @kernels
cuda_tile.module @kernels {
  // CHECK-LABEL: entry @test_func()
  entry @test_func() {
    // CHECK: constant <i32: 1> : tile<i32>
    // CHECK-NOT: loc("/tmp/foo.py"
    %c1 = constant <i32: 1> : !cuda_tile.tile<i32> loc("/tmp/foo.py":5:6)
    // CHECK: constant <i32: 2> : tile<i32>
    %c2 = constant <i32: 2> : !cuda_tile.tile<i32> loc(#di_loc)
    return loc(unknown)
  } loc(#di_loc)
} loc(unknown)
