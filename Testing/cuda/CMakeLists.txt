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
cuda_add_executable(DAXCudaTest
  main.cu
  ArgumentsParser.cxx
  ArgumentsParser.h
)

target_link_libraries(DAXCudaTest
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )


#-----------------------------------------------------------------------------
add_test(DAXCudaTest1-128
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 1 --size 128)
add_test(DAXCudaTest2-128
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 2 --size 128)
add_test(DAXCudaTest3-128
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 3 --size 128)
add_test(DAXCudaTest1-256
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 1 --size 256)
add_test(DAXCudaTest2-256
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 2 --size 256)
add_test(DAXCudaTest3-256
  ${EXECUTABLE_OUTPUT_PATH}/DAXCudaTest --pipeline 3 --size 256)
