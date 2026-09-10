// Backward-compat regression: bit-packed non-splat `tile<NxI1>` payloads
// must read and write byte identically across the v13.3 / v13.4
// version boundary.
//
// The v13.3 / v13.4 split is delicate because the splat encoding shifted
// (one-byte 0xff -> bit-packed) but the non-splat encoding did not:
// non-splats were already bit-packed in v13.3.
//
// The committed `i1-non-splat-legacy-13.3.tileirbc` was baked by the
// release/tileir-13.3 binary; the patterns inside it
// are deliberately:
//   * `i1_alt_4`        : [T,F,T,F] -- alternating, 1 byte (0x05).
//   * `i1_ff_tt`        : [F,F,T,T] -- visually obvious asymmetry,
//                                      catches bit-reversal (0x0c vs 0x03).
//   * `i1_endpoints_16` : [T, F*14, T] -- 16 elements, two bytes
//                                         (0x01 0x80), so byte ordering
//                                         is also exercised: bit 0 must
//                                         land in byte 0 and bit 15 in
//                                         byte 1.
//
// RUN-1 verifies master can read the v13.3 bytes back into the correct
// values. RUN-2 verifies master targeting v13.3 writes byte-identical
// bytes.

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc | FileCheck %s

// RUN: cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module \
// RUN:   %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:   -bytecode-version=13.3 %t.mlir -o %t.bc
// RUN: cmp %t.bc %S/Inputs/13.3/i1-non-splat-legacy-13.3.tileirbc

// CHECK-LABEL: cuda_tile.module @kernels
// CHECK:   entry @i1_alt_4
// CHECK:     constant <i1: [true, false, true, false]> : tile<4xi1>
// CHECK:   entry @i1_ff_tt
// CHECK:     constant <i1: [false, false, true, true]> : tile<4xi1>
// CHECK:   entry @i1_endpoints_16
// CHECK:     constant <i1: [true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true]> : tile<16xi1>
