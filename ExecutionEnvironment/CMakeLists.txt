#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#===========================================================================
set(sources
  DaxArray.cu
  DaxCell.cu
  DaxCommon.h
  DaxDataObject.cu
  DaxField.cu
  DaxWork.cu
  )


CUDA_ADD_LIBRARY(DaxExecutionEnvironment
  ${sources}
  SHARED)

CUDA_ADD_EXECUTABLE(DaxExecutionEnvironmentTest test.cu)
TARGET_LINK_LIBRARIES(DaxExecutionEnvironmentTest
  DaxExecutionEnvironment)
