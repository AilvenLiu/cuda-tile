//===- Types.td - CUDA Tile Type Definitions ---------------*- tablegen -*-===//
//
// Part of the CUDA Tile IR project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
#define CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

include "cuda_tile/Dialect/CudaTile/IR/AttrDefs.td"
include "cuda_tile/Dialect/CudaTile/IR/Dialect.td"
include "cuda_tile/Dialect/CudaTile/IR/Interfaces.td"

//===----------------------------------------------------------------------===//
// Versioned Type Alias
//===----------------------------------------------------------------------===//

/// CudaTile type alias with version information.
class CudaTileTypeAlias<Type t, string ver, string summary = t.summary>
  : TypeAlias<t, summary> {
  string sinceVersion = ver;
}

//===----------------------------------------------------------------------===//
// Integer Types
//===----------------------------------------------------------------------===//

// i1 values are interpreted based on operation semantics:
// - Unsigned interpretation: 0, 1 (i.e., 0b00000000, 0b00000001)
// - Signed interpretation: 0, -1 (two's complement for 1-bit, i.e., 0b00000000, 0b11111111)
// Operations on i1 values must preserve the LSB-only semantics. i1 values are
// canonicalized to 0x00 (false) or 0x01 (true) before storage and after loading
// from memory.
def CudaTile_Int1  : CudaTileTypeAlias<I1, "13.1", "i1">;
def CudaTile_Int4  : CudaTileTypeAlias<I<4>, "13.3", "i4">;
def CudaTile_Int8  : CudaTileTypeAlias<I8, "13.1", "i8">;
def CudaTile_Int16 : CudaTileTypeAlias<I16, "13.1", "i16">;
def CudaTile_Int32 : CudaTileTypeAlias<I32, "13.1", "i32">;
def CudaTile_Int64 : CudaTileTypeAlias<I64, "13.1", "i64">;

def CudaTile_AnyInt : AnyTypeOf<[CudaTile_Int1,
                                 CudaTile_Int8,
                                 CudaTile_Int16,
                                 CudaTile_Int32,
                                 CudaTile_Int64]> {
  let cppFunctionName = "isAnyInt";
}

//===----------------------------------------------------------------------===//
// Floating-point Types
//===----------------------------------------------------------------------===//

def CudaTile_Float16  : CudaTileTypeAlias<F16, "13.1", "f16">;
def CudaTile_BFloat16 : CudaTileTypeAlias<BF16, "13.1", "bf16">;
def CudaTile_Float32  : CudaTileTypeAlias<F32, "13.1", "f32">;
def CudaTile_TFloat32 : CudaTileTypeAlias<TF32, "13.1", "tf32">;
def CudaTile_Float64  : CudaTileTypeAlias<F64, "13.1", "f64">;

def CudaTile_Float8E4M3FN : CudaTileTypeAlias<F8E4M3FN, "13.1", "f8E4M3FN">;
def CudaTile_Float8E5M2   : CudaTileTypeAlias<F8E5M2, "13.1", "f8E5M2">;
def CudaTile_Float8E8M0FNU : CudaTileTypeAlias<F8E8M0FNU, "13.2", "f8E8M0FNU">;
def CudaTile_Float4E2M1FN : CudaTileTypeAlias<F4E2M1FN, "13.3", "f4E2M1FN">;
def CudaTile_Float8E5M3FNU : CudaTileTypeAlias<F8E5M3FNU, "13.4", "f8E5M3FNU">;
def CudaTile_AnyFloat : AnyTypeOf<[CudaTile_Float16,
                                   CudaTile_BFloat16,
                                   CudaTile_Float32,
                                   CudaTile_TFloat32,
                                   CudaTile_Float64,
                                   CudaTile_Float8E4M3FN,
                                   CudaTile_Float8E5M2,
                                   CudaTile_Float8E8M0FNU,
                                   CudaTile_Float4E2M1FN,
                                   CudaTile_Float8E5M3FNU,
                                  ]> {
  let cppFunctionName = "isAnyFloat";
}

def CudaTile_NumberType : AnyTypeOf<[CudaTile_AnyFloat,
                                     CudaTile_AnyInt]> {
  string cppType = "::mlir::Type";
}

//===----------------------------------------------------------------------===//
// Pointer Type
//===----------------------------------------------------------------------===//

def CudaTile_PointerType : CudaTileTypeDef<"Pointer", "ptr", "pointerType", "13.1"> {
  let summary = "Pointer type";

  let description = [{
    An element pointer type $pointerType represents a single location in
    global device memory. Pointer types are typed, i.e., they carry the
    type they point to. Any numeric type can be used as pointee type.

    An optional :code:`#cuda_tile.ptr_attr<...>` suffix records the pointer
    classification; :code:`#cuda_tile.ptr_attr<none>` denotes an ordinary local
    pointer.
  }]
  ;

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType), [{
      return $_get(pointeeType.getContext(), pointeeType, ::mlir::cuda_tile::PtrAttrAttr{});
    }]>,
    TypeBuilderWithInferredContext<(ins "Type":$pointeeType, "::mlir::cuda_tile::PtrAttrAttr":$ptrAttr), [{
      return $_get(pointeeType.getContext(), pointeeType, ptrAttr);
    }]>
  ];

  let parameters = (ins
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "13.1">:$pointeeType,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PtrAttrAttr", "ptr_attr">, "13.4">:$ptrAttr
  );

  let assemblyFormat = "`<` custom<CudaTileType>($pointeeType) (`,` $ptrAttr^)? `>`";
}

//===----------------------------------------------------------------------===//
// Tile Type
//===----------------------------------------------------------------------===//

def CudaTile_TileElementType : AnyTypeOf<[CudaTile_NumberType,
                                          CudaTile_PointerType,
                                          CudaTile_Int4
                                         ]> {
  string cppType = "::mlir::Type";
}

def CudaTile_TileType : CudaTileTypeDef<"Tile", "tile", "tileType", "13.1",
    [ShapedTypeInterface]> {
  let summary = "Tile type";

  let description = [{
    A tile is a value type that has a shape and an element type. The shape of the tile
    must be fully static. All elements of the tile have the same element
    type. Any numeric type or pointer type can be used as an
    element type.

    Only power-of-two shape dimensions are supported.

    Examples:
    ```
    !cuda_tile.tile<8x4xf32>

    !cuda_tile.tile<4x!cuda_tile.ptr<i8>>
    ```
  }];

  let parameters = (ins
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "shape of the tile">, "13.1">:$shape,
    CudaTileConstrainedTypeParam<CudaTile_TileElementType, "13.1">:$elementType
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType)>
  ];

  let extraClassDeclaration = [{
    // All interface methods of ShapedTypeInterface must be implemented.

    /// Return "true" if the type has a rank.
    bool hasRank() const { return true; }

    /// Return a new type with the given shape and element type.
    TileType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                         Type elementType) const;
  }];
}

// Checks if a type is an instance of cuda_tile::TileType
def CudaTile_IsTileTypePred
  : CPred<"::llvm::isa<::mlir::cuda_tile::TileType>($_self)">;

class CudaTile_TileOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tile">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([CudaTile_IsTileTypePred], preds)>,
      summary, "::mlir::cuda_tile::TileType"> {
        list<Type> allowedElementTypes = allowedTypes;
      }

// Ranked Tile
class CudaTile_RankedTileOf<list<Type> allowedTypes, list<int> ranks>
  : CudaTile_TileOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tile">;

// Rank-0 (Scalar) Tile
class CudaTile_ScalarTileOf<Type elementType>
  : CudaTile_RankedTileOf<[elementType], [0]>,
    BuildableType<!if(!eq(elementType.builderCall, ""), "",
      "::mlir::cuda_tile::TileType::get(ArrayRef<int64_t>(), " #
      elementType.builderCall #  ")")
    >;

//===----------------------------------------------------------------------===//
// TensorView Type
//===----------------------------------------------------------------------===//

def CudaTile_TensorViewType : CudaTileTypeDef<
    "TensorView",
    "tensor_view",
    "tensor_viewType",
    "13.1"
> {
  let summary = "tensor view type";

  let description = [{
    :code:`!cuda_tile.tensor_view` represents a reference to a tensor in global
    memory.

    It consists of:
    * :code:`elementType`: the type of the elements in the :code:`tensor_view`.
    * :code:`shape`: an integer array that specifies the size of each dimension.
      Sizes must be strictly positive.
    * :code:`strides`: an integer array that describes the stride of each
      dimension. The stride is the number of elements to offset in memory when
      increasing the corresponding index by one. Strides must be strictly
      positive.

    The shape and the stride can be dynamic on a per-dimension basis. In those
    cases, their values are printed as :code:`?`.

    .. note::

      Only power-of-two tile dimensions are supported.

    .. note::

      4-bit element types are packed densely in global memory. Tensor views with
      4-bit element types must have at least one dimension with a unit stride
      and such dimensions must have an even number of elements to ensure
      byte-alignment.

      Elements :code:`i` and :code:`i+1` (where :code:`i` is a multiple of 2)
      are packed into a byte as follows: element :code:`i` is stored in bits
      :code:`3...0` and element :code:`i+1` is stored in bits :code:`7...4`.
      This corresponds to a little-endian nibble order.

      Example: A :code:`tensor_view<2xf4E2M1FN, strides=[1]>` with values
      :code:`[0.5, 1.5]` is stored as:
      ```
      Bit position:    7     6     5     4     3     2     1     0
                    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
                    │  0  │  0  │  1  │  1  │  0  │  0  │  0  │  1  │
                    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                    ├──── upper nibble ────┤ ├──── lower nibble ────┤
                    │          1.5         │ │         0.5          │
                    └──────────────────────┘ └──────────────────────┘
      Bit pattern of 0.5 in f4E2M1FN: 0001
      Bit pattern of 1.5 in f4E2M1FN: 0011
      ```

    Examples:

    ```
    // A 512x1024 global memory tensor in row-major (lexicographic) order.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1024, 1]>

    // A 512x1024 global memory tensor in column-major (colexicographic) order.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1, 512]>

    // A 512x1024 global memory tensor that enumerates the same memory location
    // multiple times.
    !cuda_tile.tensor_view<512x1024xf16, strides=[1, 1]>

    // A 32x16x32 global memory tensor that is neither row-major nor
    // column-major.
    !cuda_tile.tensor_view<32x16x32xf16, strides=[512, 1, 16]>

    // A ?x? global memory tensor with a unit stride at the last dimension.
    !cuda_tile.tensor_view<?x?xf16, strides=[?, 1]>

    // A ?x16 global memory tensor with a unit stride at the first dimension.
    !cuda_tile.tensor_view<?x16xf32, strides=[1, ?]>
    ```
  }];

  let parameters = (ins
    CudaTileConstrainedTypeParam<CudaTile_NumberType, "13.1">:$elementType,
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "shape of the tensor view">, "13.1">:$shape,
    CudaTileTypeParam<ArrayRefParameter<"int64_t", "strides of the tensor view">, "13.1">:$strides,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PtrAttrAttr">, "13.4">:$ptrAttr
  );

  let builders = [
    TypeBuilder<(ins "Type":$elementType,
                     "ArrayRef<int64_t>":$shape,
                     "ArrayRef<int64_t>":$strides), [{
      return $_get($_ctxt, elementType, shape, strides,
                   ::mlir::cuda_tile::PtrAttrAttr{});
    }]>
  ];

  let extraClassDeclaration = [{
    /// Value used to represent dynamic shape and stride dimensions.
    static constexpr int64_t kDynamic = ::mlir::ShapedType::kDynamic;

    /// Return how many shape dimensions are dynamic.
    size_t dynamicShapeAmount();
    /// Return how many stride dimensions are dynamic.
    size_t dynamicStrideAmount();
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// PartitionView Type
//===----------------------------------------------------------------------===//

def CudaTile_PartitionViewType : CudaTileTypeDef<
      "PartitionView",
      "partition_view",
      "partitionView",
      "13.1",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "partition view type";

  let description = [{
    :code:`!cuda_tile.partition_view` represents a view into a
    :code:`tensor_view` where tiles are laid out in a grid pattern across the
    original :code:`tensor_view`. The grid is aligned with the start of each
    dimension and there are no gaps or overlaps between tiles.

    :code:`!cuda_tile.partition_view` has the following specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: a dense integer array that describes the shape of the
      tiles in the view.
    * :code:`tensor_view`: the type of the :code:`tensor_view` into which the
      view is looking.
    * :code:`dim_map`: an integer array that specifies for each tile dimension
      the corresponding dimension in the underlying :code:`tensor_view`.
    * :code:`padding_value`: an optional enum, specifying the value that should
      be used for out-of-bounds accesses (loads) into the :code:`tensor_view`.

    Supported padding values include:
    * :code:`zero`: zero
    * :code:`neg_zero`: negative zero
    * :code:`nan`: NaN
    * :code:`pos_inf`: positive infinity
    * :code:`neg_inf`: negative infinity

    .. note::

      Only power-of-two tile dimensions are supported.

    Examples:

    ```
    // (1) A view into a 16xf32 tensor_view with a tile size of 2. The table
    // below visualizes for each element of the tensor_view the corresponding
    // tile, as indicated by its index.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) (1) (1) (2) (2) (3) (3) (4) (4) (5) (5) (6) (6) (7) (7)
    //
    !pv_1d = !cuda_tile.partition_view<
      tile=(2),
      tensor_view<16xf32, strides=[1]>
    >

    // (2) A view into a 64x16xf32 tensor_view with a tile size of 4x2. By
    // convention, in the below table, the Y axis corresponds to the first
    // tensor_view dimension and the X axis corresponds to the second one.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //     │ (0,0) (0,0) (0,1) (0,1) (0,2) (0,2) (0,3) (0,3) (0,4) (0,4) ...
    //  64 │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (1,0) (1,0) (1,1) (1,1) (1,2) (1,2) (1,3) (1,3) (1,4) (1,4) ...
    //     │ (2,0) (2,0) (2,1) (2,1) (2,2) (2,2) (2,3) (2,3) (2,4) (2,4) ...
    //    ...
    //
    !pv_2d = !cuda_tile.partition_view<
      tile=(4x2),
      tensor_view<64x16xf32, strides=[16, 1]>
    >

    // (3) A view into a 64x16xf32 tensor_view with a tile size of 4x2. The
    // first tile dimension is mapped to the second tensor_view dimension. The
    // second tile dimension is mapped to the first tensor_view dimension.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //  64 │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //    ...
    //
    !pv_2d_transposed = !cuda_tile.partition_view<
      tile=(4x2),
      tensor_view<64x16xf32, strides=[16, 1]>,
      dim_map=[1, 0]
    >

    // Note: A load from partition_view with non-default dim_map is
    // semantically identical to a load with default dim_map followed by a
    // permutation.
    //
    // %0 = load_view_tko ... %view[%a, %b]
    //     : partition_view<tile=(4x2), ..., dim_map=[1, 0]> -> tile<4x2xf32>
    //
    // Is identical to:
    //
    // %0 = load_view_tko ... %view[%b, %a]
    //     : partition_view<tile=(2x4), ..., dim_map=[0, 1]> -> tile<2x4xf32>
    // %1 = permute %0 [1, 0] : tile<2x4xf32> -> tile<4x2xf32>
    ```

    The partition view index space is determined by the :code:`tile_shape`, the
    :code:`tensor_view` shape and :code:`dim_map`. In the above examples,
    :code:`!pv_2d` has an index space shape of :code:`16x8`, whereas
    :code:`!pv_2d_transposed` has an index space shape of :code:`4x32`.

    Indices into the partition view must lie within the index space of the
    partition view. Otherwise, the behavior is undefined. For example, loading
    the tile at index :code:`(0, 8)` from a partition view of type :code:`!pv_2d` above
    is invalid, because the maximum index in dimension 1 is 7.

    While partition view indices must be in-bounds, the accessed tile itself may run
    out-of-bounds. I.e., it may only partially overlap with the underlying
    :code:`tensor_view`. Tiles cannot be fully outside of the underlying
    :code:`tensor_view` because that would require the partition view indices
    to lie outside the partition view index space.
    When tiles are partially out of bounds, operations follow the following semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds tile
      elements yield the padding value. If not set, out-of-bounds elements yield
      unspecified values.
    * **Store operations**: Out-of-bounds tile elements are masked during stores.

    Example:

    ```
    // (4) A view into a 8x2xf32 tensor_view with a tile size of 1x4 and NaN
    // padding. The right half of the below table consists of padded NaN
    // values.
    //
    //            2
    //       ←─────────→
    //     ↑ (0,0) (0,0) (0,0) (0,0)
    //     │ (1,0) (1,0) (1,0) (1,0)
    //   8 │ (2,0) (2,0) (2,0) (2,0)
    //     │ (3,0) (3,0) (3,0) (3,0)
    //     │ (4,0) (4,0) (4,0) (4,0)
    //    ...
    //
    !pv_2d_padded = !cuda_tile.partition_view<
      tile=(1x4),
      padding_value = nan,
      tensor_view<8x2xf32, strides=[2,1]>,
    >
    ```
  }];

  let parameters = (ins
    CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.1">:$tile_shape,
    CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.1">:$tensor_view,
    CudaTileTypeParam<ArrayRefParameter<"int32_t", "dimension mapping">, "13.1">:$dim_map,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.1">:$padding_value
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// StridedView Type
//===----------------------------------------------------------------------===//

def CudaTile_StridedViewType : CudaTileTypeDef<
      "StridedView",
      "strided_view",
      "stridedView",
      "13.3",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "strided view type";

  let description = [{
    :code:`!cuda_tile.strided_view` represents a view into a
    :code:`tensor_view` where tiles are laid out in a grid pattern across the
    original :code:`tensor_view`. The grid is aligned with the start of each
    dimension, but the grid pattern's striding factor is
    parametric, allowing interleaved or overlapping tiles.

    :code:`!cuda_tile.strided_view` has the following specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: an integer array that describes the shape of the
      tiles in the view.
    * :code:`traversal_strides`: an integer array that describes the traversal
      strides when traversing the underlying :code:`tensor_view`. For example, if the shape of
      tiles is 2x2 and the traversal strides are [3,3], the :code:`tensor_view` will
      be traversed with a gap of one element between each tile. Alternatively,
      if the traversal strides are [1,1], the :code:`tensor_view` will be traversed
      with an overlapping sliding window, advancing by one element for each
      tile.
    * :code:`tensor_view`: the type of the :code:`tensor_view` into which the
      view is looking.
    * :code:`dim_map`: an integer array that specifies for each tile dimension
      the corresponding dimension in the underlying :code:`tensor_view`.
    * :code:`padding_value`: an optional enum, specifying the value that should
      be used for out-of-bounds accesses (loads) into the :code:`tensor_view`.

    Supported padding values include:
    * :code:`zero`: zero
    * :code:`neg_zero`: negative zero
    * :code:`nan`: NaN
    * :code:`pos_inf`: positive infinity
    * :code:`neg_inf`: negative infinity

    .. note::

      Only power-of-two tile dimensions are supported. In contrast, traversal
      strides can be any strictly positive value.

    Examples:

    ```
    // (1) A view into a 16xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 2. This behavior is identical to PartitionView. The
    // table below visualizes for each element of the tensor_view the
    // corresponding tile, as indicated by its index.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) (1) (1) (2) (2) (3) (3) (4) (4) (5) (5) (6) (6) (7) (7)
    //
    !sv_1d_tra2 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[2],
      tensor_view<16xf32, strides=[1]>
    >

    // (2) A view into a 16xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 3. This time, one out of three elements are
    // skipped, as the stride moves three elements while the tile only
    // covers two. Notice that tile 5 is partially out of bounds.
    //
    //                               16
    // ←─────────────────────────────────────────────────────────────→
    // (0) (0) ( ) (1) (1) ( ) (2) (2) ( ) (3) (3) ( ) (4) (4) ( ) (5) (5)
    //
    !sv_1d_tra3 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[3],
      tensor_view<16xf32, strides=[1]>
    >

    // (3) A view into a 8xf32 tensor_view with a tile size of 2 and a
    // traversal stride of 1. This time, tiles are overlapping, and the
    // same element will be present in multiple tiles. For example,
    // element 1 of the tensor view is present both as the second element of
    // tile 0, and as the first element tile 1. Tile 7 is partially out of bounds.
    // (Note the space is still 1-dimensional; the second row indicates overlap,
    // not a second dimension.)
    // 
    //               8
    // ←─────────────────────────────→
    // (0) (0) (2) (2) (4) (4) (6) (6)
    //     (1) (1) (3) (3) (5) (5) (7) (7)
    //
    !sv_1d_tra1 = !cuda_tile.strided_view<
      tile=(2),
      traversal_strides=[1],
      tensor_view<8xf32, strides=[1]>
    >

    // (4) A view into a 64x16xf32 tensor_view with a tile size of 4x2 and
    // traversal strides of [4,3]. Tiles are adjacent in the first dimension,
    // but have a gap of one element in the second dimension. By convention,
    // in the below table, the Y axis corresponds to the first tensor_view
    // dimension and the X axis corresponds to the second one.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //     │ (0,0) (0,0) (   ) (0,1) (0,1) (   ) (0,2) (0,2) (   ) (0,3) ...
    //  64 │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (1,0) (1,0) (   ) (1,1) (1,1) (   ) (1,2) (1,2) (   ) (1,3) ...
    //     │ (2,0) (2,0) (   ) (2,1) (2,1) (   ) (2,2) (2,2) (   ) (2,3) ...
    //    ...
    //
    !sv_2d = !cuda_tile.strided_view<
      tile=(4x2),
      traversal_strides=[4,3]
      tensor_view<64x16xf32, strides=[16, 1]>
    >

    // (5) A view into a 64x16xf32 tensor_view with a tile size of 4x2 and
    // traversal strides of [4,3], as above, but with dim_map=[1, 0].
    // (The first tile dimension is mapped to the second tensor_view dimension
    // and the second tile dimension is mapped to the first tensor_view dimension.)
    // Tiles are adjacent in the second dimension, but have a gap of one element in
    // the first dimension.
    //
    //                                   16
    //       ←────────────────────────────────────────────────────────── ...
    //     ↑ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     │ (0,0) (0,0) (0,0) (0,0) (1,0) (1,0) (1,0) (1,0) (2,0) (2,0) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //     │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //  64 │ (0,1) (0,1) (0,1) (0,1) (1,1) (1,1) (1,1) (1,1) (2,1) (2,1) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     │ (0,2) (0,2) (0,2) (0,2) (1,2) (1,2) (1,2) (1,2) (2,2) (2,2) ...
    //     | (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) (   ) ...
    //    ...
    //
    !sv_2d_transposed = !cuda_tile.strided_view<
      tile=(4x2),
      traversal_strides=[4,3],
      tensor_view<64x16xf32, strides=[16, 1]>,
      dim_map=[1, 0]
    >

    // Note: A load from a strided_view with non-default dim_map is
    // semantically identical to a load with default dim_map followed by a
    // permutation.
    //
    // %0 = load_view_tko ... %view[%a, %b]
    //     : strided_view<tile=(4x2), traversal_strides=[4,2], ...,
    //                    dim_map=[1, 0]> -> tile<4x2xf32>
    //
    // Is identical to:
    //
    // %0 = load_view_tko ... %view[%b, %a]
    //     : strided_view<tile=(2x4), traversal_strides=[2,4], ...,
    //                    dim_map=[0, 1]> -> tile<2x4xf32>
    // %1 = permute %0 [1, 0] : tile<2x4xf32> -> tile<4x2xf32>
    ```

    The strided view index space is determined by the
    :code:`traversal_strides`, the :code:`tensor_view` shape, and
    :code:`dim_map`. Partial tiles at the edges of the :code:`tensor_view`
    are included in the index space. In the above examples, :code:`!sv_1d_tra2`
    has an index space shape of :code:`8`, whereas :code:`!sv_1d_tra3` has an
    index space shape of :code:`6`. Similarly, :code:`!sv_2d` has an index
    space shape of :code:`16x6`, and :code:`!sv_2d_transposed` has an index
    space shape of :code:`4x22`.

    Indices accessing the strided view must lie within the index space of the
    strided view. Otherwise, the behavior is undefined. For example, loading
    the tile at index :code:`(0, 6)` from a strided view of type :code:`!sv_2d`
    above is invalid, as the maximum index in dimension 1 is 5.

    While strided view indices must be in-bounds, the accessed tile itself may
    run out-of-bounds. I.e., it may only partially overlap with the underlying
    :code:`tensor_view`. Tiles cannot be fully outside of the underlying
    :code:`tensor_view` because that would require the strided view indices
    to lie outside of the strided view index space.
    When tiles are partially out of bounds, operations follow the following semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds tile
      elements yield the padding value. If not set, out-of-bounds elements yield
      unspecified values.
    * **Store operations**: Out-of-bounds tile elements are masked during stores.

    Example:

    ```
    // (6) A view into a 8x2xf32 tensor_view with a tile size of 1x4 and NaN
    // padding. The right half of the below table consists of padded NaN
    // values.
    //
    //            2
    //       ←─────────→
    //     ↑ (0,0) (0,0) (0,0) (0,0)
    //     │ (1,0) (1,0) (1,0) (1,0)
    //   8 │ (2,0) (2,0) (2,0) (2,0)
    //     │ (3,0) (3,0) (3,0) (3,0)
    //     │ (4,0) (4,0) (4,0) (4,0)
    //    ...
    //
    !sv_2d_padded = !cuda_tile.strided_view<
      tile=(1x4),
      traversal_strides=[1,4],
      padding_value = nan,
      tensor_view<8x2xf32, strides=[2,1]>,
    >
    ```
  }];

  let parameters = (ins CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.3">:$tile_shape,
                        CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "traversal strides", "13.3">:$traversal_strides,
                        CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.3">:$tensor_view,
                        CudaTileTypeParam<ArrayRefParameter<"int32_t", "dimension mapping">, "13.3">:$dim_map,
                        CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.3">:$padding_value);

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// GatherScatterView Type
//===----------------------------------------------------------------------===//

def CudaTile_GatherScatterViewType : CudaTileTypeDef<
      "GatherScatterView",
      "gather_scatter_view",
      "gatherScatterView",
      "13.3",
      [DeclareTypeInterfaceMethods<CudaTile_TileView>]
> {
  let summary = "gather/scatter view type";

  let description = [{
    :code:`!cuda_tile.gather_scatter_view` represents a view into a
    :code:`tensor_view` where one dimension is accessed using a sparse
    gather/scatter pattern while the remaining dimensions are accessed
    contiguously.

    :code:`!cuda_tile.gather_scatter_view` has the following specification:
    * Index space rank: as many dimensions as the underlying
      :code:`tensor_view`.
    * Tile sizes: as specified by :code:`tile_shape`.

    It consists of:
    * :code:`tile_shape`: an integer array that describes the shape of the
      tiles in the view.
    * :code:`tensor_view`: the type of the :code:`tensor_view` into which the
      view is looking.
    * :code:`sparse_dim`: a non-negative integer that specifies which dimension
      to gather/scatter over. Must be strictly less than the rank of the
      :code:`tensor_view`.
    * :code:`padding_value`: an optional enum, specifying the value that should
      be used for out-of-bounds accesses (loads) into the :code:`tensor_view`.

    Supported padding values include:
    * :code:`zero`: zero
    * :code:`neg_zero`: negative zero
    * :code:`nan`: NaN
    * :code:`pos_inf`: positive infinity
    * :code:`neg_inf`: negative infinity

    .. note::

      Only power-of-two tile dimensions are supported.

    When loading or storing via a gather_scatter_view, the index at position
    :code:`sparse_dim` must be a **1D tile** whose size equals
    :code:`tile_shape[sparse_dim]`. Each element of this tile is an
    independent index into the underlying :code:`tensor_view` along that
    dimension. The loaded or stored tile will be the concatenation of the
    elements in the rows selected by the sparse dimension index. All other
    indices must be **scalar tiles** of the same element type, each choosing
    the offset in the underlying :code:`tensor_view` along that dimension
    and selecting a contiguous block of :code:`tile_shape[dim]` elements.

    Examples:

    ```
    // (1) A 1D gather/scatter view over an 8xf32 tensor_view with a tile
    // size of 4 and sparse_dim=0. Since the view is 1D, the only index is
    // the gather index: a size-4 1D tile of element indices. The table below
    // visualizes a load with gather_indices=[6, 1, 4, 3]. Gathered elements
    // are indicated with their index in the single loaded tile. ( ) marks
    // elements that are not gathered.
    //
    //               8
    // ←─────────────────────────────→
    // ( ) (1) ( ) (3) (2) ( ) (0) ( )
    //
    // In pseudocode, we can say that loading with gather_indices=[6, 1, 4, 3]
    // is equivalent to the following:
    //   result = [tensor_view[6], tensor_view[1], tensor_view[4], tensor_view[3]]
    //
    !gsv_1d = !cuda_tile.gather_scatter_view<
      tile=(4),
      tensor_view<8xf32, strides=[1]>,
      sparse_dim=0
    >

    // (2) A 2D gather/scatter view over an 8x8xf32 tensor_view with tile
    // size 4x4 and sparse_dim=0. The first index (at sparse_dim) is a size-4
    // 1-D tile of row indices; the second index is a scalar selecting a
    // block of 4 contiguous columns. The table below visualizes a load
    // with gather_indices=[5, 1, 7, 3] and col_idx=0. Each number
    // indicates the result row, and ( ) marks elements not gathered.
    //
    //                      8
    //       ←─────────────────────────────→
    //     ↑ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 0: not gathered
    //     │ (1) (1) (1) (1) ( ) ( ) ( ) ( )    row 1: → result row 1
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 2: not gathered
    //   8 │ (3) (3) (3) (3) ( ) ( ) ( ) ( )    row 3: → result row 3
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 4: not gathered
    //     │ (0) (0) (0) (0) ( ) ( ) ( ) ( )    row 5: → result row 0
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 6: not gathered
    //     ↓ (2) (2) (2) (2) ( ) ( ) ( ) ( )    row 7: → result row 2
    //
    !gsv_2d = !cuda_tile.gather_scatter_view<
      tile=(4x4),
      tensor_view<8x8xf32, strides=[8, 1]>,
      sparse_dim=0
    >

    // (3) A larger gather/scatter view with zero-padding for out-of-bounds
    // accesses. If the index for dimension 1 is in the range [241, 255],
    // the out-of-bounds elements will be filled with zero. If no padding value
    // were set, the values would be unspecified. Out of bounds sparse dimension
    // indices, in dimension 0, will yield a row of zero (the padding value).
    // Likewise, the values of the entire row are unspecified if no padding
    // value is set.
    //
    !gsv_2d_padded = !cuda_tile.gather_scatter_view<
      tile=(8x16),
      padding_value = zero,
      tensor_view<128x256xf32, strides=[256, 1]>,
      sparse_dim=0
    >

    // (4) Gather along the second dimension (sparse_dim=1). Here the
    // first index is a scalar selecting which block of 8 rows, and
    // the second index is a 1D tile of 16 column indices.
    //
    !gsv_2d_col = !cuda_tile.gather_scatter_view<
      tile=(8x16),
      tensor_view<128x256xf32, strides=[256, 1]>,
      sparse_dim=1
    >

    // (5) Scatter (store) using the same view type as example (2).
    // Given a 4x4 tile to store and scatter_indices=[5, 1, 7, 3]
    // with col_idx=0, row 0 of the tile is written to row 5 of the
    // tensor_view, row 1 to row 1, row 2 to row 7, and row 3 to
    // row 3. Columns 0-3 are written contiguously. The table below
    // shows which tensor_view rows are written [*] and which are
    // untouched ( ).
    //
    //                      8
    //       ←─────────────────────────────→
    //     ↑ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 0: untouched
    //     │ [1] [1] [1] [1] ( ) ( ) ( ) ( )    row 1: ← tile row 1
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 2: untouched
    //   8 │ [3] [3] [3] [3] ( ) ( ) ( ) ( )    row 3: ← tile row 3
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 4: untouched
    //     │ [0] [0] [0] [0] ( ) ( ) ( ) ( )    row 5: ← tile row 0
    //     │ ( ) ( ) ( ) ( ) ( ) ( ) ( ) ( )    row 6: untouched
    //     ↓ [2] [2] [2] [2] ( ) ( ) ( ) ( )    row 7: ← tile row 2
    //
    // In pseudocode:
    //   tensor_view[5, 0:4] = tile[0, :]
    //   tensor_view[1, 0:4] = tile[1, :]
    //   tensor_view[7, 0:4] = tile[2, :]
    //   tensor_view[3, 0:4] = tile[3, :]
    //
    !gsv_2d_scatter = !cuda_tile.gather_scatter_view<
      tile=(4x4),
      tensor_view<8x8xf32, strides=[8, 1]>,
      sparse_dim=0
    >
    ```

    The gather/scatter view index space shape equals the shape of the
    underlying :code:`tensor_view`. In the above examples, :code:`!gsv_1d`
    has an index space shape of :code:`8`, :code:`!gsv_2d` has an index
    space shape of :code:`8x8`, and :code:`!gsv_2d_padded` has an index
    space shape of :code:`128x256`.

    Indices accessing the :code:`gather_scatter_view` on non-sparse dimensions
    must be in-bounds, though the accessed tile itself may run out-of-bounds.
    Individual gather/scatter indices (at :code:`sparse_dim`) may reference
    rows that are out-of-bounds of the underlying :code:`tensor_view`.
    When tiles are partially out of bounds, operations follow the following
    semantics:
    * **Load operations**: If :code:`padding_value` is set, out-of-bounds
      elements yield the padding value. If not set, out-of-bounds elements
      yield unspecified values.
    * **Store operations**: Out-of-bounds elements are masked during stores.

    .. note::

      Gather/Scatter sparse dimension indices do not need to be unique. Repeated
      sparse index values in a load operation result in the same row being loaded
      multiple times. Repeated sparse index values in a store operation result in
      undefined values in the repeated row.
  }];

  let parameters = (ins
    CudaTileStrTypeParam<"::mlir::DenseI32ArrayAttr", "tile shape", "13.3">:$tile_shape,
    CudaTileStrTypeParam<"::mlir::cuda_tile::TensorViewType", "tensor view", "13.3">:$tensor_view,
    CudaTileStrTypeParam<"uint32_t", "sparse dimension", "13.3">:$sparse_dim,
    CudaTileTypeParam<OptionalParameter<"::mlir::cuda_tile::PaddingValueAttr", "padding value">, "13.3">:$padding_value
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

def CudaTile_TokenType : CudaTileTypeDef<"Token", "token", "token", "13.1"> {
  let summary = "cuda tile token type";
  let description = [{
    Tokens are not runtime values. Their purpose is to explicitly represent
    ordering constraints between token-ordered operations executed within a tile.
  }];
}

//===----------------------------------------------------------------------===//
// Any Type
//===----------------------------------------------------------------------===//

def CudaTile_AnyType : AnyTypeOf<[
  CudaTile_NumberType,
  Type<CPred<"::llvm::isa<::mlir::cuda_tile::CudaTileType>($_self)">>
]>;

//===----------------------------------------------------------------------===//
// Numerical Tile Types
//===----------------------------------------------------------------------===//

def CudaTile_IntTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64
]>;

def CudaTile_IntTileInt64Type : CudaTile_TileOf<[CudaTile_Int64]>;

def CudaTile_BaseFloatTileType : CudaTile_TileOf<[
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64
]>;

def CudaTile_FloatTileType : CudaTile_TileOf<[
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2,
  CudaTile_Float8E8M0FNU, CudaTile_Float4E2M1FN,
  CudaTile_Float8E5M3FNU
]>;

def CudaTile_NumberTileType : CudaTile_TileOf<[
  CudaTile_Int1, CudaTile_Int8, CudaTile_Int16, CudaTile_Int32, CudaTile_Int64,
  CudaTile_Float16, CudaTile_BFloat16, CudaTile_Float32, CudaTile_Float64,
  CudaTile_TFloat32, CudaTile_Float8E4M3FN, CudaTile_Float8E5M2,
  CudaTile_Float8E8M0FNU, CudaTile_Float4E2M1FN,
  CudaTile_Float8E5M3FNU
]>;

def CudaTile_PointerTileType : CudaTile_TileOf<[CudaTile_PointerType]>;

#ifdef TILE_IR_INCLUDE_TESTS
include "cuda_tile/Dialect/CudaTile/IR/TestingTypes.td"
#endif // TILE_IR_INCLUDE_TESTS

#endif  // CUDATILE_DIALECT_CUDATILE_IR_TYPES_TD
