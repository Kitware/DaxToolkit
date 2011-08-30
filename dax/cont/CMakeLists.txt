#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
include_directories(${Boost_INCLUDE_DIRS})

set(sources
  DataArray.cxx
  DataArrayStructuredConnectivity.cxx
  DataArrayStructuredPoints.cxx
  DataSet.cxx
  ImageData.cxx

  internal/Object.cxx
  )

set(headers
  DataArrayIrregular.h
  DataArrayIrregular.txx
  )

add_library(daxCont SHARED ${sources} ${headers})

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

target_link_libraries(daxCont
  ${Boost_LIBRARIES}
  )

add_subdirectory(internal)
