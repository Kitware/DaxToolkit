#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
include_directories(${Boost_INCLUDE_DIRS})

set(headers
  Extent.h
  Functional.h
  Types.h
  TypeTraits.h
  )

dax_declare_headers(${headers})

add_subdirectory(internal)

add_subdirectory(exec)
add_subdirectory(cont)

if (DAX_ENABLE_THRUST)
  add_subdirectory(thrust)
endif (DAX_ENABLE_THRUST)

if (DAX_ENABLE_CUDA)
  add_subdirectory(cuda)
endif (DAX_ENABLE_CUDA)

if (DAX_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif (DAX_ENABLE_OPENMP)

set(unit_tests
  UnitTestExtent.cxx
  UnitTestTypes.cxx
  UnitTestTypeTraits.cxx
  )

dax_unit_tests(SOURCES ${unit_tests})
