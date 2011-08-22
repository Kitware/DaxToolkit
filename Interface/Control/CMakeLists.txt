#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
include_directories(${Boost_INCLUDE_DIRS})

add_library(daxCont SHARED
  DataArray.cxx
  DataArray.h
  DataArrayStructuredConnectivity.cxx
  DataArrayStructuredConnectivity.h
  DataArrayStructuredPoints.cxx
  DataArrayStructuredPoints.h
  DataSet.cxx
  DataSet.h
  ImageData.cxx
  ImageData.h
)

target_link_libraries(daxCont
  ${Boost_LIBRARIES}
  daxCoreCont)
