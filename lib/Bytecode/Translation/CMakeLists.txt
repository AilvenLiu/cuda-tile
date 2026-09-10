add_mlir_library(CudaTileBytecodeTranslation
  BytecodeTranslation.cpp

  DEPENDS
    CudaTileDialect
    CudaTileBytecodeAnalysis
    CudaTileBytecodeCommon
    CudaTileBytecodeWriter
    CudaTileBytecodeReader

  LINK_LIBS PUBLIC
    CudaTileBytecodeAnalysis
    CudaTileBytecodeCommon
    CudaTileBytecodeWriter
    CudaTileBytecodeReader
    CudaTileDialect
    MLIRIR
    MLIRPass
)
