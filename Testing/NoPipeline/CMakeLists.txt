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
cuda_add_executable(NoPipelineTest
  main.cu
  ArgumentsParser.cxx
  ArgumentsParser.h
)

target_link_libraries(NoPipelineTest
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )


#-----------------------------------------------------------------------------
add_test(NoPipelineTest1-128
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 1 --size 128)
add_test(NoPipelineTest2-128
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 2 --size 128)
add_test(NoPipelineTest3-128
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 3 --size 128)
add_test(NoPipelineTest1-256
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 1 --size 256)
add_test(NoPipelineTest2-256
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 2 --size 256)
add_test(NoPipelineTest3-256
  ${EXECUTABLE_OUTPUT_PATH}/NoPipelineTest --pipeline 3 --size 256)
