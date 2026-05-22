add_mlir_dialect_library(SfDialect
  SfDialect.cpp
  SfOps.cpp
  SfLowerToLinalg.cpp
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
