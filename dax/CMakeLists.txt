#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Extent.h
  Types.h
  TypeTraits.h
  VectorOperations.h
  )

dax_declare_headers(${headers})

add_subdirectory(internal)

add_subdirectory(cont)
add_subdirectory(exec)

if (DAX_ENABLE_CUDA)
  add_subdirectory(cuda)
endif (DAX_ENABLE_CUDA)

set(unit_tests
  UnitTestExtent.cxx
  UnitTestTypes.cxx
  UnitTestTypeTraits.cxx
  UnitTestVectorOperations.cxx
  )

dax_unit_tests(SOURCES ${unit_tests})
