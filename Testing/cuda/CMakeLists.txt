#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
#CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
cuda_add_executable(DaxCudaTest
  DaxCudaTest.cu
  ArgumentsParser.cxx
  Timer.cxx
)

target_link_libraries(DaxCudaTest
  ${Boost_LIBRARIES}
  )

#-----------------------------------------------------------------------------
add_test(DaxCudaTest1-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 1 --size 128)
add_test(DaxCudaTest2-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 2 --size 128)
add_test(DaxCudaTest3-128
  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 3 --size 128)
#add_test(DaxCudaTest1-256
#  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 1 --size 256)
#add_test(DaxCudaTest2-256
#  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 2 --size 256)
#add_test(DaxCudaTest3-256
#  ${EXECUTABLE_OUTPUT_PATH}/DaxCudaTest --pipeline 3 --size 256)
