// RUN: cuda-tile-opt %s -verify-diagnostics -allow-unregistered-dialect -split-input-file

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same number of operands and results}}
    %0:2 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
      : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
      (%iter_arg : !cuda_tile.tile<f32>, %prev_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg, %prev_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{'cuda_tile.reduce' op region #0 ('body') failed to verify constraint: region with 1 blocks}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    () {}
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{custom op 'cuda_tile.reduce' number of operands and types do not match: got 0 operands and 1 types}}
    %0 = cuda_tile.reduce dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect identities to match the number of operands but got: 1 operands and 2 identities}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg : !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect 0-rank tile type at index: 0 but got: '!cuda_tile.tile<1xf32>'}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : !cuda_tile.tile<1xf32>, %prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect 0-rank tile type at index: 0 but got: 'f32'}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : f32, %prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same element type for block argument at index: 0 and 1 but got: 'f32' and 'i32'}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : !cuda_tile.tile<f32>, %prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same element type for block argument at index: 2 and 3 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.reduce %arg0, %arg1 dim=0 identities=[0.000000e+0 : f32, 0 : i32] : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
      -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
             (%arg0_iter_arg : !cuda_tile.tile<f32>,
              %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
              %arg1_iter_arg : !cuda_tile.tile<i32>,
              %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same element type for block argument at index: 0 and 1 but got: 'f32' and 'i32'}}
    %0 = cuda_tile.reduce %arg0 dim=0 identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>
    (%iter_arg : !cuda_tile.tile<f32>, %prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xi32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same type for operand at index: 0 and block argument at index: 0 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.reduce %arg0, %arg1 dim=0 identities=[0 : i32, 0.000000e+0 : f32]
        : !cuda_tile.tile<8xi32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<i32>, !cuda_tile.tile<f32>
        (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
         %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect number of terminators operands (0) to match number of operands (2)}}
    %0:2 = cuda_tile.reduce %arg0, %arg1 dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
        : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 0 and terminator argument at index: 0 but got: 'f32' and 'i32'}}
    %0:2 = cuda_tile.reduce %arg0, %arg1 dim=0 identities=[0.000000e+0 : f32, 0 : i32]
        : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg1_iter_arg, %arg0_iter_arg : !cuda_tile.tile<i32>, !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{requires the same shape for all operands}}
    %0:2 = cuda_tile.reduce %arg0, %arg1 dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
        : !cuda_tile.tile<8xf32>, !cuda_tile.tile<16xi32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{'cuda_tile.reduce' op inferred type(s) '!cuda_tile.tile<f32>', '!cuda_tile.tile<i32>' are incompatible with return type(s) of operation '!cuda_tile.tile<1xf32>', '!cuda_tile.tile<i32>'}}
    // expected-error @below{{failed to infer returned types}}
    %0:2 = cuda_tile.reduce %arg0, %arg1
      dim=0 identities=[0.000000e+0 : f32, 0 : i32]
      : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<1xf32>, !cuda_tile.tile<i32>
      (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
        %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
        cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
      }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 1 and identity at index: 1 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.reduce %arg0, %arg1
    dim=0 identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{attribute 'dim' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
    %0:2 = cuda_tile.reduce %arg0, %arg1
    dim=-10 identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @reduce_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{'cuda_tile.reduce' op dimension (10) is out of bound [0, 1)}}
    %0:2 = cuda_tile.reduce %arg0, %arg1
    dim=10 identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same number of operands and results}}
    %0:2 = cuda_tile.scan %arg0
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32>
    (%iter_arg : !cuda_tile.tile<f32>, %prev_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg, %prev_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect 2 block arguments but got: 0}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    () {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{custom op 'cuda_tile.scan' number of operands and types do not match: got 0 operands and 1 types}}
    %0 = cuda_tile.scan dim=0 reverse=false identities=[0 : i32, 0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect identities to match the number of operands but got: 1 operands and 2 identities}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield %iter_arg : !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect 0-rank tile type at index: 0 but got: '!cuda_tile.tile<1xf32>'}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%iter_arg : !cuda_tile.tile<1xf32>, %prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect 0-rank tile type at index: 0 but got: 'f32'}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%iter_arg : f32, %prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same element type for block argument at index: 0 and 1 but got: 'f32' and 'i32'}}
    %0 = cuda_tile.scan %arg0 dim=0 reverse=false identities=[0.000000e+0 : f32] : !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>
    (%iter_arg : !cuda_tile.tile<f32>, %prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same element type for block argument at index: 2 and 3 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xi32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect same type for operand at index: 0 and block argument at index: 0 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0 : i32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xi32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xi32>, !cuda_tile.tile<8xf32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xf32>) {
    // expected-error @below{{expect number of terminators operands (0) to match number of operands (2)}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xf32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<f32>, %arg1_prev_iter_arg : !cuda_tile.tile<f32>) {
      cuda_tile.yield
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 0 and terminator argument at index: 0 but got: 'f32' and 'i32'}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg1_iter_arg, %arg0_iter_arg : !cuda_tile.tile<i32>, !cuda_tile.tile<f32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<16xi32>) {
    // expected-error @below{{requires the same shape for all operands}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<16xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<16xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----


cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 0 and result at index: 0}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<16xf32>, !cuda_tile.tile<16xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{expect same type for operand at index: 1 and identity at index: 1 but got: 'i32' and 'f32'}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=0 reverse=false identities=[0.000000e+0 : f32, 0.000000e+0 : f32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{attribute 'dim' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=-10 reverse=false identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  testing$func @scan_operation(%arg0: !cuda_tile.tile<8xf32>, %arg1: !cuda_tile.tile<8xi32>) {
    // expected-error @below{{'cuda_tile.scan' op dimension (10) is out of bound [0, 1)}}
    %0:2 = cuda_tile.scan %arg0, %arg1
    dim=10 reverse=false identities=[0.000000e+0 : f32, 0 : i32]
    : !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32> -> !cuda_tile.tile<8xf32>, !cuda_tile.tile<8xi32>
    (%arg0_iter_arg : !cuda_tile.tile<f32>, %arg0_prev_iter_arg : !cuda_tile.tile<f32>,
     %arg1_iter_arg : !cuda_tile.tile<i32>, %arg1_prev_iter_arg : !cuda_tile.tile<i32>) {
      cuda_tile.yield %arg0_iter_arg, %arg1_iter_arg : !cuda_tile.tile<f32>, !cuda_tile.tile<i32>
    }
  }
}

// -----

cuda_tile.module @kernels {
  entry @reduce_kernel() {
    %cst_f32 = constant <f32: [1.000000e+00, -2.000000e+00, 3.000000e+00, -4.000000e+00, 5.000000e+00, -6.000000e+00, 7.000000e+00, -8.000000e+00]> : tile<8xf32>
    %cst_f32_0 = constant <f32: 5.000000e-01> : tile<8xf32>
    %cst_2_f32 = constant <f32: 2.000000e+00> : tile<8xf32>
    %0:3 = reduce %cst_f32, %cst_f32_0, %cst_2_f32 dim=0 identities=[1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32] : tile<8xf32>, tile<8xf32>, tile<8xf32> -> tile<f32>, tile<f32>, tile<f32>
    (%arg0: tile<f32>, %arg1: tile<f32>, %arg2: tile<f32>, %arg3: tile<f32>, %arg4: tile<f32>, %arg5: tile<f32>) {
      %2 = mulf %arg0, %arg1  : tile<f32>
      %3 = mulf %arg2, %arg3  : tile<f32>
      %4 = mulf %arg4, %arg5  : tile<f32>
      // expected-error @below{{'cuda_tile.print_tko' op only pure operations are allowed inside 'cuda_tile.reduce'}}
      %5 = print_tko "reduce_step" -> token
      yield %2, %3, %4 : tile<f32>, tile<f32>, tile<f32>
    }
    %1 = print_tko "reduce_results %f %f %f", %0#0, %0#1, %0#2 : tile<f32>, tile<f32>, tile<f32> -> token
    return
  }
}

// -----

cuda_tile.module @kernels {
  entry @scan_kernel() {
    %cst_f32 = constant <f32: [1.000000e+00, -2.000000e+00, 3.000000e+00, -4.000000e+00, 5.000000e+00, -6.000000e+00, 7.000000e+00, -8.000000e+00]> : tile<8xf32>
    %cst_f32_0 = constant <f32: 5.000000e-01> : tile<8xf32>
    %cst_2_f32 = constant <f32: 2.000000e+00> : tile<8xf32>
    %0:3 = scan %cst_f32, %cst_f32_0, %cst_2_f32 dim=0 reverse=true identities=[1.000000e+00 : f32, 2.000000e+00 : f32, 3.000000e+00 : f32] : tile<8xf32>, tile<8xf32>, tile<8xf32> -> tile<8xf32>, tile<8xf32>, tile<8xf32>
    (%arg0: tile<f32>, %arg1: tile<f32>, %arg2: tile<f32>, %arg3: tile<f32>, %arg4: tile<f32>, %arg5: tile<f32>) {
      %2 = mulf %arg0, %arg1  : tile<f32>
      %3 = mulf %arg2, %arg3  : tile<f32>
      %4 = mulf %arg4, %arg5  : tile<f32>
      // expected-error @below{{'cuda_tile.print_tko' op only pure operations are allowed inside 'cuda_tile.scan'}}
      %5 = print_tko "scan_step" -> token
      yield %2, %3, %4 : tile<f32>, tile<f32>, tile<f32>
    }
    %1 = print_tko "scan_results %f %f %f", %0#0, %0#1, %0#2 : tile<8xf32>, tile<8xf32>, tile<8xf32> -> token
    return
  }
}
