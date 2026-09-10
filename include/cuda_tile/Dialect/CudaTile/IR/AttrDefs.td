//===- AttrDefs.td - CUDA Tile Attribute Definitions -------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD
#define CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD

include "mlir/IR/EnumAttr.td"

include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// Pointer attributes
//===----------------------------------------------------------------------===//

def CudaTile_PtrAttrEnum : CudaTileI32EnumAttr<
    "PtrAttr", "pointer classification", "13.4",
    [
        CudaTileI32EnumAttrCase<"Default local pointer", "NONE", 0, "13.4", "none">,
    ]> {
  let specPrefixDescription = [{
    The :code:`ptr_attr` attribute classifies :code:`!cuda_tile.ptr` values.
    :code:`none` denotes an ordinary local pointer. It is a distinct spelling
    from carrying no :code:`ptr_attr` at all: ops that relate a pointer to a
    view require the two to agree on whether the attribute is present.
  }];
  let specSuffixDescription = "";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_PtrAttr : CudaTileEnumAttr<CudaTile_PtrAttrEnum, "ptr_attr"> {
  let summary = "Pointer classification";
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

#ifdef TILE_IR_INCLUDE_TESTS
include "cuda_tile/Dialect/CudaTile/IR/TestingAttrs.td"
#endif // TILE_IR_INCLUDE_TESTS

//===----------------------------------------------------------------------===//
// MLIR Builtin Attribute Aliases with Version Information
// These define version information for MLIR builtin attrs used in bytecode.
//===----------------------------------------------------------------------===//

class CudaTileAttrAlias<string name, string version> {
  string attrName = name;
  string sinceVersion = version;
}

def CudaTile_IntegerAttr : CudaTileAttrAlias<"Integer", "13.1">;
def CudaTile_FloatAttr : CudaTileAttrAlias<"Float", "13.1">;
def CudaTile_BoolAttr : CudaTileAttrAlias<"Bool", "13.1">;
def CudaTile_TypeAttr : CudaTileAttrAlias<"Type", "13.1">;
def CudaTile_StringAttr : CudaTileAttrAlias<"String", "13.1">;
def CudaTile_ArrayAttr : CudaTileAttrAlias<"Array", "13.1">;
def CudaTile_DenseElementsAttr : CudaTileAttrAlias<"DenseElements", "13.1">;
def CudaTile_DictionaryAttr : CudaTileAttrAlias<"Dictionary", "13.1">;

//===----------------------------------------------------------------------===//
// Integer Signedness Attribute
//===----------------------------------------------------------------------===//

def CudaTile_Signedness : CudaTileI32EnumAttr<"Signedness", "signedness", "13.1",
    [CudaTileI32EnumAttrCase<"Treat the operands as unsigned integers.", "Unsigned", 0, "13.1", "unsigned">,
     CudaTileI32EnumAttrCase<"Treat the operands as signed integers.", "Signed", 1, "13.1", "signed">]> {
  let specPrefixDescription = "The :code:`signedness` attribute specifies the signedness of operand(s).";
  let specSuffixDescription = "";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_SignednessAttr : CudaTileEnumAttr<CudaTile_Signedness, "signedness"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// Integer Overflow Attributes
//===----------------------------------------------------------------------===//

def CudaTile_IntegerOverflow : CudaTileI32EnumAttr<
  "IntegerOverflow", "integer overflow", "13.1",
  [
      CudaTileI32EnumAttrCase<"The compiler makes no assumptions regarding overflow behavior.", "NONE", 0, "13.1", "none">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands signed integers.", "NSW", 1, "13.1", "no_signed_wrap">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands unsigned integers.", "NUW", 2, "13.1", "no_unsigned_wrap">,
      CudaTileI32EnumAttrCase<"The compiler assumes that overflow (wrap-around) will not occur when interpreting the operands as signed or unsigned integers.", "NW", 3, "13.1", "no_wrap">,
  ]> {
  let specPrefixDescription = [{
    The :code:`overflow` attribute is used to instruct the compiler on how to reason about the overflow behavior of the specific operation.

    These attributes serve as assumptions that the compiler may use to reason about the operation. It is the responsibility of the code generator to ensure that the operation
    respects these assumptions dynamically during execution.
  }];
  let specSuffixDescription = "If an overflow occurs at runtime despite the value of overflow stating otherwise, the behavior is undefined.";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}


def CudaTile_IntegerOverflowAttr :
    CudaTileEnumAttr<CudaTile_IntegerOverflow, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// GPU Architecture (single source of truth for all architecture lists)
//===----------------------------------------------------------------------===//
def CudaTile_GpuArchitecture : CudaTileI32EnumAttr<
    "GpuArchitecture", "gpu architecture", "13.1",
    [CudaTileGpuArchCase<"SM 8.0",  "SM_80",  80,  "13.2", "sm_80",  "Ampere">,
     CudaTileGpuArchCase<"SM 8.6",  "SM_86",  86,  "13.2", "sm_86",  "Ampere">,
     CudaTileGpuArchCase<"SM 8.7",  "SM_87",  87,  "13.2", "sm_87",  "Ampere">,
     CudaTileGpuArchCase<"SM 8.8",  "SM_88",  88,  "13.2", "sm_88",  "Ampere">,
     CudaTileGpuArchCase<"SM 8.9",  "SM_89",  89,  "13.2", "sm_89",  "Ada">,
     CudaTileGpuArchCase<"SM 9.0",  "SM_90",  90,  "13.3", "sm_90",  "Hopper">,
     CudaTileGpuArchCase<"SM 10.0", "SM_100", 100, "13.1", "sm_100", "Blackwell">,
     CudaTileGpuArchCase<"SM 10.3", "SM_103", 103, "13.1", "sm_103", "Blackwell">,
     CudaTileGpuArchCase<"SM 10.7", "SM_107", 107, "13.4", "sm_107", "Rubin">,
     CudaTileGpuArchCase<"SM 11.0", "SM_110", 110, "13.1", "sm_110", "Blackwell">,
     CudaTileGpuArchCase<"SM 12.0", "SM_120", 120, "13.1", "sm_120", "Blackwell">,
     CudaTileGpuArchCase<"SM 12.1", "SM_121", 121, "13.1", "sm_121", "Blackwell">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// Optimization Hint Key
//===----------------------------------------------------------------------===//

// Extends CudaTileI32EnumAttrCase with metadata for auto-generating
// getter methods, validators, verifier dispatch, and op-applicability
// on OptimizationHintsAttr.
//
// validationKind: "int_range", "pow2_range", "bool", or "string"
// validMin/validMax: inclusive range for int_range/pow2_range
// validForOps: list of op class names this hint applies to (for verifyWithOp)
// archOverrides: per-architecture-family range narrowing (via HintArchOverride)
class CudaTileHintKeyCase<string desc, string sym, int val, string version,
                          string str, string retType,
                          string validationKind,
                          int validMin = 0, int validMax = 0,
                          list<string> validForOps = [],
                          list<HintArchOverride> archOverrides = [],
                          string docDescription = "">
    : CudaTileI32EnumAttrCase<desc, sym, val, version, str> {
  string hintReturnType = retType;
  string hintAccessorName = sym;
  string hintValidationKind = validationKind;
  int hintValidMin = validMin;
  int hintValidMax = validMax;
  list<string> hintValidForOps = validForOps;
  list<HintArchOverride> hintArchOverrides = archOverrides;
  string hintDocDescription = docDescription;
}


def CudaTile_HintKey : CudaTileI32EnumAttr<
    "HintKey", "optimization hint key", "13.1",
    [CudaTileHintKeyCase<"Number of CTAs in CGA.",
         "NumCTAInCGA", 0, "13.1", "num_cta_in_cga",
         "int", "pow2_range", 1, 16, ["EntryOp"],
         [HintArchOverride<["Ampere", "Ada"], 1, 1>],
         "suggest the number of CTAs in a CGA">,
     CudaTileHintKeyCase<"Occupancy hint.",
         "Occupancy", 1, "13.1", "occupancy",
         "int", "int_range", 1, 32, ["EntryOp"],
         [],
         "occupancy hint">,
     CudaTileHintKeyCase<"Allow TMA usage.",
         "AllowTMA", 2, "13.1", "allow_tma",
         "bool", "bool", 0, 0, ["LoadViewTkoOp", "StoreViewTkoOp"],
         [],
         "suggest whether to use TMA">,
     CudaTileHintKeyCase<"Latency hint.",
         "Latency", 3, "13.1", "latency",
         "int", "int_range", 1, 10, ["LoadViewTkoOp", "StoreViewTkoOp", "LoadPtrTkoOp", "StorePtrTkoOp"],
         [],
         "latency hint">,
     CudaTileHintKeyCase<"Number of worker warps per CTA. "
         "Controls how many warps are allocated to the worker (non-specialized) "
         "agent in a warp-specialized kernel.  After specialization the final "
         "CTA may contain additional warps for MMA, TMA, or control agents, so "
         "this value does not represent the total warp count.",
         "NumWorkerWarpsPerCTA", 4, "13.3", "num_worker_warps_per_cta",
         "int", "pow2_range", 4, 8, ["EntryOp"],
         [],
         "suggest the number of worker warps per CTA (excluding "
         "specialized MMA/TMA/control warps)">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

//===----------------------------------------------------------------------===//
// Optimization hints Attributes
//===----------------------------------------------------------------------===//

def CudaTile_OptimizationHintsAttr : CudaTileAttrDef<"OptimizationHints", "optimization_hints", "13.1"> {
  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"DictionaryAttr", "optimization hints dictionary">, "13.1">:$value);
  // The per-hint documentation (bullet list and op-to-hint table) is
  // auto-generated by the spec generator from CudaTile_HintKey metadata
  // (hintDocDescription, validationKind, validMin/Max, validForOps,
  // archOverrides).  Only the intro paragraph is written here.
  let description = [{
    The :code:`optimization_hints` attribute provides architecture-specific compiler hints in the form of nested dictionaries.

    The hints are specified for each architecture (e.g., :code:`sm_100`, :code:`sm_120`) and for each architecture the user can specify
    specific hints for each operation.

    User can specify :code:`default` hint that will be used when no target-specific hint is provided
    (and only when :code:`default` hint value is in the valid range for the current compilation target).

    For example they can be annotated as:

    .. code-block:: mlir

      optimization_hints=<
        sm_100 = {num_cta_in_cga = 8, num_worker_warps_per_cta = 8},
        sm_120 = {num_cta_in_cga = 16, num_worker_warps_per_cta = 4}
      >
  }];
  let hasCustomAssemblyFormat = 1;
  let cppNamespace = "::mlir::cuda_tile";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{

  private:
    static constexpr llvm::StringLiteral kDefault = "default";

    static bool isKnownKey(llvm::StringRef key) {
      return key == kDefault || symbolizeGpuArchitecture(key).has_value();
    }

    mlir::LogicalResult verifyParamWithContext(Location loc,
                                               llvm::StringRef context,
                                               ArrayRef<StringRef> keysValidForOperation,
                                               DictionaryAttr &attr);
  public:
}] #
// Auto-generated from CudaTileHintKeyCase definitions.
// Provides kXxx() string accessors and getXxx(StringRef sm) getters.
!interleave(!foreach(case, CudaTile_HintKey.enumerants,
    "\n    static llvm::StringRef k" # case.symbol
    # "() { return stringifyHintKey(HintKey::" # case.symbol # "); }"
    # "\n    std::optional<" # case.hintReturnType
    # "> get" # case.symbol
    # "(StringRef sm, mlir::Operation *op = nullptr);"
), "") # [{

    mlir::LogicalResult verifyWithOp(Operation *op, DictionaryAttr value);

  }];
}

//===----------------------------------------------------------------------===//
// Rounding Mode Attributes
//===----------------------------------------------------------------------===//

def CudaTile_RoundingMode : CudaTileI32EnumAttr<
  "RoundingMode", "rounding mode", "13.1",
  [   CudaTileI32EnumAttrCase<"Round to nearest (ties to even).", "NEAREST_EVEN", 0, "13.1", "nearest_even">,
      CudaTileI32EnumAttrCase<"Round towards zero (truncate).", "ZERO", 1, "13.1", "zero">,
      CudaTileI32EnumAttrCase<"Round towards negative infinity.", "NEGATIVE_INF", 2, "13.1", "negative_inf">,
      CudaTileI32EnumAttrCase<"Round towards positive infinity.", "POSITIVE_INF", 3, "13.1", "positive_inf">,
      CudaTileI32EnumAttrCase<"Approximate rounding mode.", "APPROX", 4, "13.1", "approx">,
      CudaTileI32EnumAttrCase<"Full precision rounding mode.", "FULL", 5, "13.1", "full">,

      // Integer roundings
      CudaTileI32EnumAttrCase<"Round towards zero to the nearest integer.", "NEAREST_INT_TO_ZERO", 6, "13.1", "nearest_int_to_zero">,

      // Float roundings
      CudaTileI32EnumAttrCase<"Round to nearest (ties away from zero).", "NEAREST_AWAY", 7, "13.4", "nearest_away">
  ]> {
  let specPrefixDescription = "The :code:`rounding` attribute specifies the rounding mode to use for the operation.";
  let specSuffixDescription = "";
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_RoundingModeAttr : CudaTileEnumAttr<CudaTile_RoundingMode, "rounding"> {
  let assemblyFormat = "`<` $value `>`";
}




//===----------------------------------------------------------------------===//
// Comparison Attributes
//===----------------------------------------------------------------------===//

def CudaTile_ComparisonOrdering : CudaTileI32EnumAttr<"ComparisonOrdering", "comparison_ordering", "13.1",
    [CudaTileI32EnumAttrCase<"Unordered comparison.", "UNORDERED", 0, "13.1", "unordered">,
     CudaTileI32EnumAttrCase<"Ordered comparison.", "ORDERED", 1, "13.1", "ordered">]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
  let specPrefixDescription = "The :code:`comparison_ordering` attribute specifies the kind of ordering to be performed in the comparison operation.";
  let specSuffixDescription = "";
}

def CudaTile_ComparisonOrderingAttr : CudaTileEnumAttr<CudaTile_ComparisonOrdering, "comparison_ordering"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_ComparisonPredicate : CudaTileI32EnumAttr<
    "ComparisonPredicate", "cmp_predicate", "13.1",
    [
      CudaTileI32EnumAttrCase<"Equal comparison.", "EQUAL", 0, "13.1", "equal">,
      CudaTileI32EnumAttrCase<"Not equal comparison.", "NOT_EQUAL", 1, "13.1", "not_equal">,
      CudaTileI32EnumAttrCase<"Less than comparison.", "LESS_THAN", 2, "13.1", "less_than">,
      CudaTileI32EnumAttrCase<"Less than or equal comparison.", "LESS_THAN_OR_EQUAL", 3, "13.1", "less_than_or_equal">,
      CudaTileI32EnumAttrCase<"Greater than comparison.", "GREATER_THAN", 4, "13.1", "greater_than">,
      CudaTileI32EnumAttrCase<"Greater than or equal comparison.", "GREATER_THAN_OR_EQUAL", 5, "13.1", "greater_than_or_equal">
    ]> {
  let cppNamespace = "::mlir::cuda_tile";
  let genSpecializedAttr = 0;
  let specPrefixDescription = "The :code:`comparison_predicate` attribute specifies the kind of comparison to be performed.";
  let specSuffixDescription = "";
}

def CudaTile_ComparisonPredicateAttr : CudaTileEnumAttr<CudaTile_ComparisonPredicate, "comparison_predicate"> {
  let assemblyFormat = "`<` $value `>`";
  let cppNamespace = "::mlir::cuda_tile";
}



//===----------------------------------------------------------------------===//
// Op-specific Attributes
//===----------------------------------------------------------------------===//

def CudaTile_AtomicRMWModeAttr : CudaTileI32EnumAttr<
    "AtomicRMWMode", "", "13.1",
    [
      CudaTileI32EnumAttrCase<"Perform bitwise AND as the modification operation.", "AND", 0, "13.1", "and">,
      CudaTileI32EnumAttrCase<"Perform bitwise OR as the modification operation.", "OR", 1, "13.1", "or">,
      CudaTileI32EnumAttrCase<"Perform bitwise XOR as the modification operation.", "XOR", 2, "13.1", "xor">,
      CudaTileI32EnumAttrCase<"Perform integer addition as the modification operation.", "ADD", 3, "13.1", "add">,
      CudaTileI32EnumAttrCase<"Perform floating-point addition as the modification operation.", "ADDF", 4, "13.1", "addf">,
      CudaTileI32EnumAttrCase<"Perform maximum as the modification operation.", "MAX", 5, "13.1", "max">,
      CudaTileI32EnumAttrCase<"Perform minimum as the modification operation.", "MIN", 6, "13.1", "min">,
      CudaTileI32EnumAttrCase<"Perform unsigned maximum as the modification operation.", "UMAX", 7, "13.1", "umax">,
      CudaTileI32EnumAttrCase<"Perform unsigned minimum as the modification operation.", "UMIN", 8, "13.1", "umin">,
      CudaTileI32EnumAttrCase<"Perform exchange as the modification operation.", "XCHG", 9, "13.1", "xchg">
    ]> {
  let specPrefixDescription = "The :code:`mode` attribute specifies the mode of the atomic read-modify-write operation.";
  let specSuffixDescription = "The :code:`mode` attribute has a default value of :code:`add`.";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_DivByAttr : CudaTileAttrDef<"DivBy", "div_by", "13.1",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {

  let description = [{
    .. code-block:: mlir

      div_by< $divisor (, every $every^ along $along)?>

    The :code:`div_by` attribute must be used as a predicate for :code:`cuda_tile.assume`
    ops. The predicated value must be a :code:`tile` of integers or pointers, or
    a :code:`tensor_view`.

    If the predicated value is a :code:`tile`, the attribute indicates that some
    elements of the :code:`tile` are divisible by :code:`divisor`. If the predicated value
    is a :code:`tensor_view` the attribute indicates that the base address of the :code:`tensor_view` is
    divisible by :code:`divisor`. :code:`divisor` must be a positive power of :code:`2`.

    The :code:`every` and :code:`along` attributes control which elements are assumed to
    satisfy the divisibility property. When splitting the tensor in groups of
    size :code:`every` along dimension :code:`along`, the first element of each group is
    assumed to satisfy the divisibility property. The other elements are
    assumed to be monotonically increasing by :code:`1` within the group. In case
    of a :code:`tile` of pointers, the elements are assumed to be monotonically
    increasing by the byte width of the pointee type. The size of the last
    group may be smaller than :code:`every`.

    The :code:`every` and :code:`along` attributes are optional. When missing, they are
    assumed to have a default value of :code:`1` and :code:`0` in case of a :code:`tile`.
    I.e., all elements of the :code:`tile` are assumed to satisfy the divisibility
    property. (The value of :code:`along` does not matter in that case.) If the
    predicated value is a :code:`tensor_view` or a 0D :code:`tile`, :code:`every` and :code:`along` cannot be
    used.

    :code:`every`, and :code:`along` must be used together. If one is specified,
    so must be the other.

    .. note::

      If the predicated value is a tile of integers, :code:`every` is a property of
      the signed interpretation of the integer values. Otherwise, it is a
      property of the unsigned integer interpretation. E.g., :code:`every = 4`
      is incorrect for the following sequence of "i8" values (written in binary
      form) because they wrap around when interpreted as signed integers:
      :code:`[01111110, 01111111, 10000000, 10000001]`. :code:`every = 2` would
      be correct.

    The examples below demonstrate tensors that satisfy the assumed properties.
  }];

  let mlirExamples = [
    [{
      // Example 1: Each pointer is divisible by 16.
      // [ 0x10, 0x20, 0x80, 0x10, 0x0, 0x120, ... ]
      %0 = cuda_tile.assume #cuda_tile.div_by<16>, %ptrs
          : !cuda_tile.tile<128x!cuda_tile.ptr<f32>>
      // Note: Equivalent to #cuda_tile.div_by<16, every 1 along 0>.
    }],
    [{
    // Example 2: Each integer is divisible by 4.
    // [ 16, 24, 8, 4, 12, 12, 0, 16, ... ]
    %0 = cuda_tile.assume #cuda_tile.div_by<4>, %t
        : !cuda_tile.tile<128xi32>
    }],
    [{
    // Example 3: Group size [4].
    // [7, 8, 9, 10, 23, 24, 25, 26, 0, 1, 2, 3, ...]
    %0 = cuda_tile.assume #cuda_tile.div_by<1, every 4 along 0>, %t
        : !cuda_tile.tile<128xi32>
    }],
    [{
    // Example 4: 2-d Group size [1, 4] with divisibility 4.
    // [ [  4,  5,  6,  7, 12, 13, 14, 15 ],
    //   [  8,  9, 10, 11, 24, 25, 26, 27 ],
    //   [ 24, 25, 26, 27, 64, 65, 66, 67 ],
    //   [  0,  1,  2,  3,  4,  5,  6,  7 ] ]
    %0 = cuda_tile.assume #cuda_tile.div_by<4, every 4 along 1>, %t
        : !cuda_tile.tile<4x8xi32>
    }],
    [{
    // Example 5: 2-d Group size [4, 1] with divisibility 32.
    // Note that the elements within each column are monotonically increasing
    // by the byte width of the pointee type f32, e.g., 0x20, 0x24, 0x28, 0x2c.
    // [ [  0x20, 0x100,  0x40,  0x60,  0x40, 0x200, 0x340,  0x40 ],
    //   [  0x24, 0x104,  0x44,  0x64,  0x44, 0x204, 0x344,  0x44 ],
    //   [  0x28, 0x108,  0x48,  0x68,  0x48, 0x208, 0x348,  0x48 ],
    //   [  0x2c, 0x10c,  0x4c,  0x6c,  0x4c, 0x20c, 0x34c,  0x4c ] ]
    %0 = cuda_tile.assume #cuda_tile.div_by<32, every 4 along 0>, %ptrs
        : !cuda_tile.tile<4x8x!cuda_tile.ptr<f32>>
    }]
  ];


  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"uint64_t", "divisor value">, "13.1">:$divisor,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>", "elements per group">, "13.1">:$every,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>", "dimension index">, "13.1">:$along);

  // TODO: Specify assembly format instead of hand-written parsers/printers.
  // This requires a fix in MLIR. Optional type parameters are not supported
  // at the moment.
  let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `<` $divisor (`,` `every` $every^ `along` $along)? `>`";
  // }];
}

def CudaTile_SameElementsAttr : CudaTileAttrDef<
    "SameElements", "same_elements", "13.1",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {
  let description = [{
    .. code-block:: mlir

      #same_elements< $values >

    The :code:`same_elements` attribute must be used as a predicate for
    :code:`cuda_tile.assume`. The predicated value must be a tensor of integers or
    pointers.

    :code:`same_elements` is specified for each dimension. A value of C for a
    dimension of size N indicates that, after dividing the respective
    dimension into N/C groups of size C, each group consists of the same
    elements. As N/C may not divide evenly, the last group may have fewer
    than C elements.

    If the "same elements" property does not hold along a dimension, the
    respective value should be set to 1.
    :code:`#cuda_tile.same_elements<[1, 1, ..., 1]>` is a correct predicate for any
    tensor of integers or pointers, where the number of ones matches the rank
    of the tensor. (Size-1 groups always have the same elements.)
  }];

  let mlirExamples = [[{
    // Integer tensor with same elements.
    %0 = cuda_tile.constant <i16: [[0, 0, 0, 0, 10, 10, 10, 10],
                                   [0, 0, 0, 0, 10, 10, 10, 10],
                                   [5, 5, 5, 5, 93, 93, 93, 93],
                                   [5, 5, 5, 5, 93, 93, 93, 93]]>
        : tile<4x8xi16>
    %1 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %0
        : !cuda_tile.tile<4x8xi16>

    // Pointer tensor with same elements.
    %2 = cuda_tile.constant <i64: [[ 0,  0,  0,  0,  8,  8,  8,  8],
                                   [ 0,  0,  0,  0,  8,  8,  8,  8],
                                   [64, 64, 64, 64, 32, 32, 32, 32],
                                   [64, 64, 64, 64, 32, 32, 32, 32]]>
        : tile<4x8xi64>
    %3 = cuda_tile.bitcast %2
        : !cuda_tile.tile<4x8xi64>
          -> !cuda_tile.tile<!cuda_tile.ptr<f32>>
    %4 = cuda_tile.assume #cuda_tile.same_elements<[2, 4]>, %3
        : !cuda_tile.tile<!cuda_tile.ptr<f32>>
  }]];

  let parameters = (ins
    CudaTileAttrParam<TypeParameter<"DenseI64ArrayAttr", "same elements values">, "13.1">:$values);
  let assemblyFormat =  "`<` $values `>`";
}

def CudaTile_BoundedAttr : CudaTileAttrDef<
    "Bounded", "bounded", "13.1",
    [DeclareAttrInterfaceMethods<CudaTile_AssumePredicateAttrInterface>]> {
  let description = [{
    .. code-block:: mlir

      #bounded<(lb|?), (ub|?)>

    The :code:`bounded` attribute must be used as a predicate for
    :code:`cuda_tile.assume`. The predicated value must be a tile of integers.

    :code:`bounded` specifies a lower and upper bound for all elements of the
    predicated tile when interpreted as signed integers. Bounds are optional:
    it is possible to leave a bound unspecified, as indicated by "?" in the
    assembly format. E.g., :code:`#bounded<0, ?>`. Both lower bound and upper
    bound are inclusive.

    The lower bounds must be less than or equal to the upper bound. A lower/
    upper bound that exceeds the range of valid values of the predicated value
    is invalid.
  }];

  let mlirExamples = [[{
    %1 = cuda_tile.assume #cuda_tile.bounded<0, ?>, %0
        : !cuda_tile.tile<4x8xi16>
  }]];

  let parameters = (ins
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>", "lower bound">, "13.1">:$lb,
    CudaTileAttrParam<OptionalParameter<"std::optional<int64_t>", "upper bound">, "13.1">:$ub);
  let assemblyFormat = [{
    `<` ($lb^) : (`?`)? `,` ($ub^) : (`?`)? `>`
  }];
}

def CudaTile_MemoryScopeAttr
    : CudaTileI32EnumAttr<"MemoryScope", "memory scope", "13.1",
                  [CudaTileI32EnumAttrCase<"There may be concurrent accesses from within the same tile block.", "TL_BLK", 0, "13.1", "tl_blk">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses from within the same device (i.e., GPU).", "DEVICE", 1, "13.1", "device">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses from anywhere within the system (i.e., all devices).", "SYS", 2, "13.1", "sys">]> {
  let specPrefixDescription = [{
    The :code:`memory_scope` attribute specifies a communication scope for memory operations.
    When communicating with other concurrent threads in the system, the scope must be broad enough to encompass all other
    threads which are participating in the communication, or data races may occur.
  }];
  let specSuffixDescription = "";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_MemoryOrderingSemanticsAttr
    : CudaTileI32EnumAttr<"MemoryOrderingSemantics", "memory ordering semantics", "13.1",
                  [CudaTileI32EnumAttrCase<"No concurrent accesses to the source/destination location.", "WEAK", 0, "13.1", "weak">,
                   CudaTileI32EnumAttrCase<"There may be concurrent access to the location, but this access does not establish a happens-before relationship.", "RELAXED", 1, "13.1", "relaxed">,
                   CudaTileI32EnumAttrCase<" There may be concurrent accesses to the location. If this acquire observes a release operation, then *happens before* is established.", "ACQUIRE", 2, "13.1", "acquire">,
                   CudaTileI32EnumAttrCase<"There may be concurrent access to the location. If this release is observed with an acquire operation, then *happens before* is established.", "RELEASE", 3, "13.1", "release">,
                   CudaTileI32EnumAttrCase<"There may be concurrent accesses to the location. This has the effect of both a release and acquire operation.", "ACQ_REL", 4, "13.1", "acq_rel">]> {
  let specPrefixDescription = [{
    The :code:`memory_ordering_semantics` attribute specifies the concurrency assumption between memory accesses in different threads, which controls the synchronization required.
    For example, :code:`weak` ordering allows the compiler to assume that there are no concurrent accesses to any accessed location.
    For more information, refer to the :ref:`memory model section <section-memory-model>` of the specification.
  }];
  let specSuffixDescription = "";
  let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_PaddingValue : CudaTileI32EnumAttr<
    "PaddingValue", "load padding value for out of bound access", "13.1",
    [
      CudaTileI32EnumAttrCase<"zero", "zero", 0, "13.1", "zero">,
      CudaTileI32EnumAttrCase<"negative zero", "neg_zero", 1, "13.1", "neg_zero">,
      CudaTileI32EnumAttrCase<"NaN", "nan", 2, "13.1", "nan">,
      CudaTileI32EnumAttrCase<"positive infinity", "pos_inf", 3, "13.1", "pos_inf">,
      CudaTileI32EnumAttrCase<"negative infinity", "neg_inf", 4, "13.1", "neg_inf">
    ]> {
    let specPrefixDescription = [{
      The :code:`padding_value` attribute specifies the value to return for an out-of-bounds access.
    }];

    let specSuffixDescription = [{
      Note that special padding values (:code:`neg_zero`, :code:`nan`, :code:`pos_inf`, :code:`neg_inf`)
      can only be used with floating-point element types.
    }];
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_PaddingValueAttr :
    CudaTileEnumAttr<CudaTile_PaddingValue, "padding_value"> {
  let cppNamespace = "::mlir::cuda_tile";
  let assemblyFormat = "$value";
}


//===----------------------------------------------------------------------===//
// DebugInfo
//===----------------------------------------------------------------------===//

/// Wrapper class for declaring CudaTile debug info attributes.
class CudaTile_DIAttr<string name, string attrMnemonic, string version,
                      list<Trait> traits = [],
                      string baseCppClass = "::mlir::Attribute">
    : AttrDef<CudaTile_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  string sinceVersion = version;
}

/// Base class for all debug info attributes.
class CudaTile_DINodeAttr<string name, string attrMnemonic, string version,
                          list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, version, traits, "DINodeAttr">;

/// Represents a debug info scope.
class CudaTile_DIScopeAttr<string name, string attrMnemonic, string version,
                           list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, version, traits, "DIScopeAttr">;

/// Represents a local debug info scope.
class CudaTile_DILocalScopeAttr<string name, string attrMnemonic, string version,
                                list<Trait> traits = []>
    : CudaTile_DIAttr<name, attrMnemonic, version, traits, "DILocalScopeAttr">;

//===----------------------------------------------------------------------===//
// DILocAttr
//===----------------------------------------------------------------------===//

def CudaTile_DILocAttr : LocationAttrDef<CudaTile_Dialect, "DILoc"> {
  let summary = "a source location with a debug info scope";
  let description = [{
    Represents a location in the source code that carries a corresponding
    debug info scope. This location is used to connect an operation with a
    particular debug scope, such as a function to its subprogram.
  }];
  let mnemonic = "di_loc";

  // Version when this attribute was introduced.
  string sinceVersion = "13.1";

  let parameters = (ins
    "FileLineColLoc":$sourceLoc,
    "DILocalScopeAttr":$scope
  );
  let assemblyFormat = "`<` $sourceLoc `in` $scope `>`";
}

//===----------------------------------------------------------------------===//
// DICompileUnitAttr
//===----------------------------------------------------------------------===//

def CudaTile_DICompileUnitAttr : CudaTile_DIScopeAttr<"DICompileUnit",
                                                      "di_compile_unit", "13.1",
                                                      /*traits=*/[]> {
  let description = [{
    Represents a compilation unit, the root scope of all objects declared
    in a specific compilation unit; specifies the associated source file
    for the compilation unit.
  }];
  let parameters = (ins
    "DIFileAttr":$file
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DIFileAttr
//===----------------------------------------------------------------------===//

def CudaTile_DIFileAttr : CudaTile_DIScopeAttr<"DIFile",
                                               "di_file", "13.1",
                                               /*traits=*/[]> {
  let description = [{
    Represents a source file; specifies the file name and directory of the
    source file.
  }];
  let parameters = (ins "StringAttr":$name, "StringAttr":$directory);
  let assemblyFormat = "`<` $name `in` $directory `>`";
}

//===----------------------------------------------------------------------===//
// DILexicalBlockAttr
//===----------------------------------------------------------------------===//

def CudaTile_DILexicalBlockAttr : CudaTile_DILocalScopeAttr<"DILexicalBlock",
                                                            "di_lexical_block", "13.1",
                                                            /*traits=*/[]> {
  let description = [{
    Represents a lexical block nested within a subprogram; specifies the
    scope, file, line number and optional column number of the block. A
    lexical block, for example, may be used to represent the nested scope
    of a conditional statement.
  }];
  let parameters = (ins
    "DILocalScopeAttr":$scope,
    "DIFileAttr":$file,
    "unsigned":$line,
    OptionalParameter<"unsigned">:$column
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// DISubprogramAttr
//===----------------------------------------------------------------------===//

def CudaTile_DISubprogramAttr : CudaTile_DILocalScopeAttr<"DISubprogram",
                                                          "di_subprogram", "13.1",
                                                          /*traits=*/[]> {
  let description = [{
    Represents a function within the source language; specifies the scope, file,
    line number, name, and linkage name of the subprogram. Optionally the line
    number within the scope can be included.
  }];
  let parameters = (ins
    "DIFileAttr":$file,
    "unsigned":$line,
    "StringAttr":$name,
    "StringAttr":$linkageName,
    "DICompileUnitAttr":$compileUnit,
    OptionalParameter<"unsigned">:$scopeLine
  );
  let assemblyFormat = "`<` struct(params) `>`";

  let extraClassDeclaration = [{
    /// Return an otherwise exact copy of `this` with a new linkage name.
    DISubprogramAttr cloneWithNewLinkageName(StringAttr newLinkageName);
  }];

  let extraClassDefinition = [{
    $cppClass $cppClass::cloneWithNewLinkageName(StringAttr newLinkageName) {
      return $cppClass::get(getContext(), getFile(), getLine(), getName(), newLinkageName,
              getCompileUnit(), getScopeLine());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Symbol Visibility Attribute
//===----------------------------------------------------------------------===//

def CudaTile_SymbolVisibility : CudaTileI32EnumAttr<
    "SymbolVisibility", "symbol visibility", "13.1",
    [CudaTileI32EnumAttrCase<"Public visibility.", "Public", 0, "13.1", "public">,
     CudaTileI32EnumAttrCase<"Private visibility.", "Private", 1, "13.1", "private">]> {
    let specPrefixDescription = "The :code:`symbol_visibility` attribute specifies the visibility of a symbol.";
    let specSuffixDescription = "";
    let genSpecializedAttr = 0;
    let cppNamespace = "::mlir::cuda_tile";
}

def CudaTile_SymbolVisibilityAttr : CudaTileEnumAttr<CudaTile_SymbolVisibility, "symbol_visibility"> {
    let assemblyFormat = "`<` $value `>`";
}

#endif  // CUDATILE_DIALECT_CUDATILE_IR_ATTRDEFS_TD
