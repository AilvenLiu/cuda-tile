add_mlir_library(CudaTileBytecodeAnalysis
  MinVersionAnalyzer.cpp

  DEPENDS
    CudaTileBytecodeIncGen
    CudaTileIncGen

  LINK_LIBS PUBLIC
    CudaTileBytecodeCommon
    CudaTileDialect
    MLIRIR
    MLIRSupport

  LINK_COMPONENTS
    Support
)

# Include Writer's build directory to access generated .inc files.
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Writer)
