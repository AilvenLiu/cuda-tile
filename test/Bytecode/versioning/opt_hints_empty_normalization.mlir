// Regression test for the parse + bytecode-reader normalization of
// `optimization_hints=<>` on `cuda_tile.entry`.
//
// `optimization_hints=<>` is semantically identical to having no
// `optimization_hints` clause at all. Both must produce the same
// in-memory IR, and therefore the same bytecode bytes.
//
// Two committed bytecodes test this from both sides:
//
//   opt-hints-empty-13.4.tileirbc
//     Baked at v13.4 with the normalization in place. Has no
//     HasOptimizationHints flag set, byte-identical to the bake of
//     a kernel that omits the `optimization_hints` clause entirely.
//     Tests the post-normalization pipeline.
//
//   opt-hints-empty-legacy-13.4.tileirbc
//     Hand-baked at v13.4 by temporarily reverting the parser fix,
//     so the bytecode has the HasOptimizationHints flag set and an
//     empty payload. Tests backward compatibility: the new reader
//     must still accept this shape, and the printer must elide the
//     empty attribute on output, so the resulting mlir text is
//     identical to what the post-normalization fixture produces.
//
// COM: Source for both fixtures (only the parser used at bake time differs):
// COM: cuda_tile.module @kernels {
// COM:   cuda_tile.entry @explicit_empty_hints() optimization_hints=<> {
// COM:     return
// COM:   }
// COM: }

// 1) Read the post-normalization bytecode. The printer must not
//    surface `optimization_hints` because the reader sees no flag
//    and stores nothing.
// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/Inputs/13.4/opt-hints-empty-13.4.tileirbc \
// RUN:     | FileCheck %s --check-prefix=POSTNORM

// POSTNORM-LABEL: cuda_tile.module @kernels
// POSTNORM:       entry @explicit_empty_hints() {
// POSTNORM-NOT:   optimization_hints

// 2) Re-bake and confirm encoding stability at v13.4 -- the new
//    code is a 1-iteration fixed point on this input.
// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/Inputs/13.4/opt-hints-empty-13.4.tileirbc -o %t.mlir
// RUN: cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module \
// RUN:     -bytecode-version=13.4 %t.mlir -o %t.tileirbc
// RUN: cmp %t.tileirbc %S/Inputs/13.4/opt-hints-empty-13.4.tileirbc

// 3) Read the LEGACY bytecode (pre-normalization payload with the
//    HasOptimizationHints flag set + empty attribute). The new
//    reader must accept it, and the printer's existing elision must
//    drop the empty attribute so the resulting text matches the
//    post-normalization output.
// RUN: cuda-tile-translate -cudatilebc-to-mlir %S/Inputs/13.4/opt-hints-empty-legacy-13.4.tileirbc \
// RUN:     | FileCheck %s --check-prefix=LEGACY

// LEGACY-LABEL: cuda_tile.module @kernels
// LEGACY:       entry @explicit_empty_hints() {
// LEGACY-NOT:   optimization_hints
