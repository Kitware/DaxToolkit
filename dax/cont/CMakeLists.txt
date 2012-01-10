#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
include_directories(${Boost_INCLUDE_DIRS})

set(sources
  internal/Object.cxx
  )

set(impl_headers
  internal/Object.h
  )

set(headers
  ArrayHandle.h
  UniformGrid.h

  ${Dax_BINARY_DIR}/dax/cont/VectorOperations.h
  )

add_subdirectory(internal)

dax_declare_headers(${impl_headers} ${headers})

#-----------------------------------------------------------------------------
set(DAX_VECTOR_OPERATIONS_NAMESPACE_NAME "dax_cont")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_START "namespace dax { namespace cont {")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_END   "} }")
set(DAX_VECTOR_OPERATIONS_EXPORT "DAX_CONT_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/VectorOperations.h.in
  ${Dax_BINARY_DIR}/dax/cont/VectorOperations.h
  @ONLY
  )

#-----------------------------------------------------------------------------
add_library(daxCont SHARED ${sources} ${headers} ${impl_headers})

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

target_link_libraries(daxCont
  ${Boost_LIBRARIES}
  )

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestArrayHandle.cxx
  UnitTestVectorOperations.cxx
  )
dax_unit_tests(SOURCES ${unit_tests} LIBRARIES daxCont)
