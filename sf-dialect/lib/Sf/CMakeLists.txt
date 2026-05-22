add_mlir_dialect_library(SfDialect
  SfDialect.cpp
  SfLowerActivation.cpp
  SfLowerAttention.cpp
  SfLowerCompare.cpp
  SfLowerGenOps.cpp
  SfLowerMatmul.cpp
  SfLowerSeqOps.cpp
  SfLowerNormalization.cpp
  SfLowerReduce.cpp
  SfLowerShape.cpp
  SfLowerToLinalg.cpp
  SfOps.cpp
  SfPromoteWeights.cpp
  SfStripGEPNuw.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Sf

  DEPENDS
    SfOpsIncGen
    SfPassesIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRSupport
    MLIRTransformUtils
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
    MLIRFuncDialect
    MLIRLinalgTransforms
    MLIRArithUtils
    MLIRAffineDialect
    MLIRSCFDialect
    MLIRLLVMDialect
)
