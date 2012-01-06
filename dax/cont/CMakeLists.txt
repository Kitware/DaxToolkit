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
  Array.h
  ArrayHandle.h
  DataSet.h
  FieldData.h
  FieldHandles.h
  StructuredGrid.h
  UniformGrid.h
  Worklets.h
  )

add_subdirectory(internal)

dax_declare_headers(${impl_headers} ${headers})

add_library(daxCont SHARED ${sources} ${headers} ${impl_headers})

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

target_link_libraries(daxCont
  ${Boost_LIBRARIES}
  )

set(unit_tests
  UnitTestArrayHandle.cxx
  )
dax_unit_tests(SOURCES ${unit_tests} LIBRARIES daxCont)
