#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#===========================================================================
project(DaxExecutionEnvironment)

include_directories(
  ${DaxCommon_SOURCE_DIR}
  ${DaxDataModel_SOURCE_DIR}
  )

set(sources
  DaxDataArrayIrregular.h
  DaxDataArrayStructuredPoints.h
  DaxDataArrayStructuredConnectivity.h
  DaxDataArrayTraits.h
  DaxCell.h
  DaxCellTypes.h
  DaxCommon.h
  DaxExecutionEnvironment.h
  DaxField.h
  DaxWork.h
  )

CUDA_ADD_LIBRARY(daxExecutionEnvironment
  ${sources}
)

target_link_libraries(daxExecutionEnvironment
  daxDataModel)

CUDA_ADD_EXECUTABLE(DaxExecutionEnvironmentTest DaxExecutionEnvironmentTest.cu
  DaxArgumentsParser.cxx
  DaxArgumentsParser.h
)

CUDA_ADD_EXECUTABLE(SimpleCUDAKernelTest SimpleCUDAKernelTest.cu
)

target_link_libraries(DaxExecutionEnvironmentTest
  daxExecutionEnvironment
  ${BOOST_LIBRARIES}
  )

target_link_libraries(SimpleCUDAKernelTest
  daxExecutionEnvironment
  )
