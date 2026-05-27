add_mlir_library(CudaTileTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  CudaTileTransforms
)
