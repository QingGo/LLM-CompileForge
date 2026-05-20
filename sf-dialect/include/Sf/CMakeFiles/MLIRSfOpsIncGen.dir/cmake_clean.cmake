file(REMOVE_RECURSE
  "CMakeFiles/MLIRSfOpsIncGen"
  "SfOps.cpp.inc"
  "SfOps.h.inc"
  "SfOpsDialect.cpp.inc"
  "SfOpsDialect.h.inc"
  "SfOpsTypes.cpp.inc"
  "SfOpsTypes.h.inc"
  "SfPasses.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRSfOpsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
