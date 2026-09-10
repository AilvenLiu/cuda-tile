// This validates that proper errors are generated when version requirements aren't met.

// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=250.0 %s -split-input-file 2>&1 | FileCheck %s --check-prefixes=CHECK-ATTR,CHECK-OPTIONAL-ATTR,CHECK-OPERAND,CHECK-RESULT,CHECK-TYPE-PARAM,CHECK-OPTIONAL-PARAM,CHECK-ENUM-ATTR,CHECK-ATTR-PARAM
// RUN: not cuda-tile-translate -mlir-to-cudatilebc -no-implicit-module -bytecode-version=13.1 %s -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-OP-NOT-AVAILABLE


// Test case 1: Attribute version error
cuda_tile.module @attribute_version_error_test {
  entry @test_attribute_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-ATTR: attribute 'new_param' requires bytecode version 250.1+, but targeting 250.0

// -----

// Test case 2: Optional attribute version error
cuda_tile.module @optional_attribute_version_error_test {
  entry @test_optional_attr_error() {
    testing$bytecode_test_new_attribute new_flag
    return
  }
}

// CHECK-OPTIONAL-ATTR: optional attribute 'new_flag' is provided but requires bytecode version 250.1, targeting 250.0

// -----

// Test case 3: Operand version error  
cuda_tile.module @operand_version_error_test {
  entry @test_operand_error() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token_in = make_token : !cuda_tile.token
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) token = %token_in : !cuda_tile.token -> !cuda_tile.token
    return
  }
}

// CHECK-OPERAND: optional operand 'optional_token' is provided but requires bytecode version 250.1, targeting 250.0

// -----

// Test case 4: Result version error
cuda_tile.module @result_version_error_test {
  entry @test_result_error() {
    %input = constant <f32: [1.0, 2.0]> : !cuda_tile.tile<2xf32>
    %token = testing$bytecode_test_evolution (%input : !cuda_tile.tile<2xf32>) -> !cuda_tile.token
    %joined = join_tokens %token, %token : !cuda_tile.token
    return
  }
}

// CHECK-RESULT: result 'result_token' requires bytecode version 250.1 but is being used and targeting 250.0

// -----

// Test case 5: Op version error
cuda_tile.module @op_version_error_test {
  entry @test_op_error() {
    testing$bytecode_test_new_attribute new_param = 123
    return
  }
}

// CHECK-OP-NOT-AVAILABLE: operation 'cuda_tile.testing$bytecode_test_new_attribute' is not available in bytecode version 13.1

// -----

// Test case 6: Type parameter version error - DefaultValuedParameter.
cuda_tile.module @type_param_explicit_default_error {
  entry @test_explicit_default_error(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, f16>) {
    return
  }
}

// CHECK-TYPE-PARAM: parameter 'newType' requires bytecode version 250.1+, but targeting 250.0

// -----

// Test case 7: Optional parameter version error.
cuda_tile.module @optional_attr_error {
  entry @test_optional_attr_error(%arg0: !cuda_tile.testing$bytecode_test_evolved<f32, i32, zero>) {
    return
  }
}

// CHECK-OPTIONAL-PARAM: parameter 'optionalAttr' requires bytecode version 250.1+, but targeting 250.0

// -----

// Test case 8: Enum value version error - 'high' case requires 250.1.
// The enum type is available in 250.0, but the 'high' value was added in 250.1.
cuda_tile.module @enum_value_version_error {
  entry @test_enum_value_version_error() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) priority = #cuda_tile.bytecode_test_priority<high> -> !cuda_tile.token
    return
  }
}

// CHECK-ENUM-ATTR: enum value for attribute 'priority' is not available in bytecode version 250.0

// -----

// Test case 9: Evolved attr parameter version error.
// BytecodeTestEvolvedAttr is available at 250.0, but its optionalValue
// parameter requires 250.1. Providing a value at 250.0 must be an error.
cuda_tile.module @evolved_attr_param_version_error {
  entry @test_evolved_attr_param_error() {
    %input = constant <f32: [1.0]> : !cuda_tile.tile<1xf32>
    %t = testing$bytecode_test_evolution (%input : !cuda_tile.tile<1xf32>) test_attr = #cuda_tile.bytecode_test_evolved_attr<42, 100> -> !cuda_tile.token
    return
  }
}

// CHECK-ATTR-PARAM: attribute parameter 'optionalValue' requires bytecode version 250.1+, but targeting 250.0
