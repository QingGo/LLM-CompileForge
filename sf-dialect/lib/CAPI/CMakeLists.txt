add_mlir_public_c_api_library(SfCAPI
  Dialects.cpp

  LINK_LIBS PUBLIC
    SfDialect
    MLIRCAPIIR
)
