file(REMOVE_RECURSE
  "libSfDialect.a"
  "libSfDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/SfDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
