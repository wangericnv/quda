include_directories(../utils)

# enable utils build a common library for all utils
add_library(
  quda_reference STATIC
  covdev_reference.cpp
  dslash_test_helpers.cpp
  hisq_force_reference.cpp
  dslash_reference.cpp
  wilson_dslash_reference.cpp
  clover_reference.cpp
  domain_wall_dslash_reference.cpp
  gauge_force_reference.cpp
  hisq_force_reference2.cpp
  staggered_dslash_reference.cpp)

# if(QUDA_QMP AND QUDA_DOWNLOAD_USQCD AND NOT QUDA_QMPHOME)
# add_dependencies(quda_reference QMP) endif()

if(QUDA_QIO
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QIO_FOUND)
  add_dependencies(quda_reference QIO)
endif()
if(QUDA_QMP
   AND QUDA_DOWNLOAD_USQCD
   AND NOT QMP_FOUND)
  add_dependencies(quda_reference QMP)
endif()
