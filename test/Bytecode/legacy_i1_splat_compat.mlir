// Backward compatibility test for the legacy i1 raw buffer encoding.
//
// Older MLIR wrote splat-true `tensor<i1>` constants as a single 0xff byte. 
// The current MLIR raw buffer contract treats only bit 0 of the splat byte as
// significant, so without normalization the legacy 0xff byte is decoded as the
// integer 255 and casts to 255.0 instead of 1.0. The reader masks the splat
// byte for i1 so legacy bytecode keeps loading correctly.
//
// The fixture `legacy_i1_splat_0xff.tileirbc` is identical to a freshly
// emitted bytecode for `<i1: 1> : tile<i1>` except that the splat byte in
// the constant section is 0xff instead of the canonical 0x01.

// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/legacy_i1_splat_0xff.tileirbc -no-implicit-module | FileCheck %s

// CHECK-LABEL: entry @i1_splat_true
// CHECK: constant <i1: true> : tile<i1>
// CHECK-NOT: <i1: 255>
