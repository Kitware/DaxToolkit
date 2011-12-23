#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
#CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
cuda_add_executable(DaxCUDATest
  DaxCUDATest.cu
  ArgumentsParser.cxx
  ArgumentsParser.h
)

target_link_libraries(DaxCUDATest
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )

#-----------------------------------------------------------------------------
add_test(DaxCUDATest1-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 1 --size 128)
add_test(DaxCUDATest2-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 2 --size 128)
add_test(DaxCUDATest3-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 3 --size 128)
add_test(DaxCUDATest1-256
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 1 --size 256)
add_test(DaxCUDATest2-256
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 2 --size 256)
add_test(DaxCUDATest3-256
  ${EXECUTABLE_OUTPUT_PATH}/DaxCUDATest --pipeline 3 --size 256)
