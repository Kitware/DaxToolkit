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
  FilterFieldTypes.h
  )

set(headers
  internal/ModuleBase.h
  internal/FilterBase.h
  internal/BaseArray.h
  DataSet.h
  Array.h
  StructuredGrid.h
  )

dax_declare_headers(${impl_headers})

add_library(daxCont SHARED ${sources} ${headers} ${impl_headers})

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

target_link_libraries(daxCont
  ${Boost_LIBRARIES}
  )

add_subdirectory(internal)
