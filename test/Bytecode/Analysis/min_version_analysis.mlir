// Tests for MinVersionAnalyzer utility.
// RUN: cuda-tile-translate -analyze-min-version -split-input-file %s | FileCheck %s

// Base operations (constant, addf, return) are available in 13.1.
// CHECK: 13.1
cuda_tile.module @basic_13_1 {
  entry @kernel() {
    %c1 = constant <f32: 1.0> : tile<f32>
    %c2 = constant <f32: 2.0> : tile<f32>
    %r = addf %c1, %c2 : tile<f32>
    return
  }
}

// -----

// The atan2 operation was introduced in 13.2.
// CHECK: 13.2
cuda_tile.module @atan2_13_2 {
  entry @kernel() {
    %c1 = constant <f32: 1.0> : tile<f32>
    %c2 = constant <f32: 2.0> : tile<f32>
    %r = atan2 %c1, %c2 : tile<f32>
    return
  }
}

// -----

// The f8E8M0FNU type was introduced in 13.2.
// CHECK: 13.2
cuda_tile.module @f8e8m0fnu_13_2 {
  entry @kernel(%p: tile<ptr<f8E8M0FNU>>) {
    return
  }
}

// -----

// The alloca operation was introduced in 13.3.
// CHECK: 13.3
cuda_tile.module @alloca_13_3 {
  entry @kernel() {
    %p = alloca num_elem = 64, alignment = 128 : tile<ptr<f32>>
    return
  }
}

// -----

// The f4E2M1FN type was introduced in 13.3.
// CHECK: 13.3
cuda_tile.module @f4e2m1fn_13_3 {
  entry @kernel(%p: tile<ptr<f4E2M1FN>>) {
    return
  }
}

// -----

// The 'unsigned' attribute on for loop was added in 13.2 (op itself is 13.1).
// CHECK: 13.2
cuda_tile.module @for_unsignedCmp_13_2 {
  entry @kernel() {
    %c0 = constant <i32: 0> : tile<i32>
    %c10 = constant <i32: 10> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    for unsigned %iv in (%c0 to %c10, step %c1) : tile<i32> {
      continue
    }
    return
  }
}

// -----

// The result_token result was added in 250.1, but is not used here.
// CHECK: 250.0
cuda_tile.module @evolution_unused_result_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    return
  }
}

// -----

// The new_attr attribute was added in 250.1 (op itself is 250.0).
// CHECK: 250.1
cuda_tile.module @evolution_new_attr_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) new_attr = 42 -> !cuda_tile.token
    return
  }
}

// -----

// The optional_token operand was added in 250.1 (op itself is 250.0).
// CHECK: 250.1
cuda_tile.module @evolution_operand_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) token = %token_in : !cuda_tile.token -> !cuda_tile.token
    return
  }
}

// -----

// The 'normal' enum value is 250.0, so version is 250.0.
// CHECK: 250.0
cuda_tile.module @evolution_enum_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<normal> -> !cuda_tile.token
    return
  }
}

// -----

// The 'high' enum value was added in 250.1.
// CHECK: 250.1
cuda_tile.module @evolution_enum_high_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<high> -> !cuda_tile.token
    return
  }
}

// -----

// The result_token is used here, so requires 250.1.
// CHECK: 250.1
cuda_tile.module @evolution_used_result_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    %joined = join_tokens %token, %token : !cuda_tile.token
    return
  }
}

// -----

// BytecodeTestEvolvedAttr without optional param: base attr version 250.0.
// CHECK: 250.0
cuda_tile.module @evolution_attr_no_optional_250_0 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123> -> !cuda_tile.token
    return
  }
}

// -----

// BytecodeTestEvolvedAttr with optional param: param version 250.1.
// CHECK: 250.1
cuda_tile.module @evolution_attr_with_optional_250_1 {
  entry @kernel() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<123, 456> -> !cuda_tile.token
    return
  }
}

// -----

// The new_flag attribute was added in 250.1 (op itself is 250.0).
// CHECK: 250.1
cuda_tile.module @new_attribute_flag_250_1 {
  entry @kernel() {
    testing$bytecode_test_new_attribute new_flag
    return
  }
}

// -----

// The new_param attribute was added in 250.1 (op itself is 250.0).
// CHECK: 250.1
cuda_tile.module @new_attribute_param_250_1 {
  entry @kernel() {
    testing$bytecode_test_new_attribute new_param = 100
    return
  }
}

// -----

// A pointer type without a ptr_attr classification remains a 13.1 feature: the
// optional attribute is absent, so it must not raise the minimum version.
// CHECK: 13.1
cuda_tile.module @plain_pointer_13_1 {
  entry @kernel(%ptr: tile<ptr<i32>>) {
    return
  }
}

