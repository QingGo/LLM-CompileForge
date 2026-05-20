file(REMOVE_RECURSE
  "libSfCAPI.a"
  "libSfCAPI.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/SfCAPI.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
