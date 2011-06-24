#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
project (DaxDataModel)

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${DaxCommon_SOURCE_DIR}
  )

set (sources
  daxDataArray.cxx
  daxDataArray.h
  daxDataArrayIrregular.h
  daxDataArrayIrregular.txx
  daxDataSet.cxx
  daxDataSet.h
  daxImageData.cxx
  daxImageData.h
  )

CUDA_ADD_LIBRARY(daxDataModel SHARED
  ${sources})

target_link_libraries(daxDataModel
  daxCommon
  ${Boost_LIBRARIES})
