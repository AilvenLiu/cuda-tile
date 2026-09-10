// Backward-compat regression: legacy v13.2-era binaries always wrote a
// 1-byte presence flag for the `padding_value` OptionalEnum on
// `cuda_tile.partition_view`. Master must produce the same bytes when
// targeting v13.2 -- including the `byte(0)` for the no-padding case.
//
// The committed `partition_view_optional_enum_legacy-13.2.tileirbc`
// was baked by the release/tileir-13.2 binary so it is the ground
// truth for v13.2 wire bytes.

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy-13.2.tileirbc | FileCheck %s

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.2/partition_view_optional_enum_legacy-13.2.tileirbc -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.2 %t.mlir -o %t.bc
// RUN: cmp %t.bc %S/Inputs/13.2/partition_view_optional_enum_legacy-13.2.tileirbc

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:   entry @partition_view_no_padding
// CHECK-SAME:    partition_view<tile=(2), tensor_view<16xf32, strides=[1]>>
// CHECK-NOT:     padding_value
// CHECK:   entry @partition_view_with_padding
// CHECK-SAME:    partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>
