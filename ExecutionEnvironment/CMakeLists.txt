#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#===========================================================================
project(DaxExecutionEnvironment)

include_directories(${DaxDataModel_SOURCE_DIR})

set(sources
  DaxArray.cu
  DaxArrayIrregular.cu
  DaxArrayStructuredConnectivity.cu
  DaxArrayStructuredPoints.cu
  DaxArrayTraits.cu
  DaxCell.cu
  DaxCellTypes.h
  DaxCommon.h
  DaxDataObject.cu
  DaxExecutionEnvironment.h
  DaxField.cu
  DaxWork.cu
  )

CUDA_ADD_EXECUTABLE(DaxExecutionEnvironmentTest DaxExecutionEnvironmentTest.cu
  ${sources}
)

