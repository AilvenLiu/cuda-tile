// Test reader type version errors.
//
// This bytecode file claims to be version 13.1 but contains Float4E2M1FN type
// which was added in 13.3. The reader should reject it with a version error.
//

// COM: The bytecode would decode to the following IR if version check was not enforced:
// COM: cuda_tile.module @kernels {
// COM:   testing$func @kernel(%arg0: tile<8xf4E2M1FN>, %arg1: tile<i32>) {
// COM:     %0 = extract %arg0[%arg1] : tile<8xf4E2M1FN> -> tile<4xf4E2M1FN>
// COM:     return
// COM:   }
// COM: }

// RUN: not cuda-tile-translate -cudatilebc-to-mlir -no-implicit-module %S/Inputs/13.1/f4e2m1fn-in-13.1.tileirbc 2>&1 | FileCheck %s

// CHECK: type 'Float4E2M1FN' requires bytecode version 13.3+, file version is 13.1

