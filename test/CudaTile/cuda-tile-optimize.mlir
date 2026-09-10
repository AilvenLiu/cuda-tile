// RUN: cuda-tile-optimize %s --quiet --enable-multithread
// RUN: cuda-tile-optimize %s --verbose --fuse-fma --run-before-default-pipeline=canonicalize --run-after-default-pipeline=cse -o %t.mlir 2>&1 | FileCheck %s --check-prefix=VERBOSE
// RUN: FileCheck %s --check-prefix=MLIR < %t.mlir
// RUN: cuda-tile-optimize %s --quiet --emit-bytecode -o %t.tileirbc
// RUN: cuda-tile-optimize %t.tileirbc --quiet
// RUN: not cuda-tile-optimize %s --quiet --run-before-default-pipeline=no-such-pass 2>&1 | FileCheck %s --check-prefix=BAD-PIPELINE
// RUN: not cuda-tile-optimize %S/Inputs/cuda-tile-optimize-invalid.txt --quiet 2>&1 | FileCheck %s --check-prefix=BAD-INPUT
// RUN: not cuda-tile-optimize %S/Inputs/cuda-tile-optimize-missing.mlir --quiet 2>&1 | FileCheck %s --check-prefix=NO-FILE

// VERBOSE: Pipeline:
// MLIR: cuda_tile.module @optimizer_test
// BAD-PIPELINE: Failed to parse pipeline: no-such-pass
// BAD-INPUT: Failed to parse input
// NO-FILE: Failed to read file:

cuda_tile.module @optimizer_test {
  cuda_tile.entry @empty() {
    cuda_tile.return
  }
}
