#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif (DAX_ENABLE_OPENMP)

if (DAX_ENABLE_CUDA)
  find_package(CUDA REQUIRED)
endif (DAX_ENABLE_CUDA)

include_directories(${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

if (DAX_ENABLE_THRUST)
  include_directories(${THRUST_INCLUDE_DIR})
endif (DAX_ENABLE_THRUST)

#-----------------------------------------------------------------------------
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/ThresholdConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/ThresholdConfig.h
  @ONLY
  )

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  #PipelineDebug.cxx
  #ThresholdTiming.cxx
  ArgumentsParser.cxx
  Timer.cxx
  )

#-----------------------------------------------------------------------------
add_executable(ThresholdTimingDebug ${sources} ${headers} PipelineDebug.cxx)
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_link_libraries(ThresholdTimingDebug
  ${Boost_LIBRARIES}
  )

add_test(ThresholdTimingDebug1-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingDebug --pipeline 1 --size 128)
add_test(ThresholdTimingDebug2-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingDebug --pipeline 2 --size 128)
add_test(ThresholdTimingDebug3-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingDebug --pipeline 3 --size 128)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(ThresholdTimingOpenMP ${sources} ${headers} PipelineOpenMP.cxx)
  target_link_libraries(ThresholdTimingOpenMP
    ${Boost_LIBRARIES}
    )
endif (DAX_ENABLE_OPENMP)

add_test(ThresholdTimingOpenMP1-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingOpenMP --pipeline 1 --size 128)
add_test(ThresholdTimingOpenMP2-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingOpenMP --pipeline 2 --size 128)
add_test(ThresholdTimingOpenMP3-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingOpenMP --pipeline 3 --size 128)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  cuda_add_executable(ThresholdTimingCuda
    ${sources} ${headers} PipelineCuda.h PipelineCuda.cu
    )
  set_source_files_properties(PipelineCuda.h PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(ThresholdTimingCuda
    ${Boost_LIBRARIES}
    )
endif (DAX_ENABLE_CUDA)

add_test(ThresholdTimingCuda1-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingCuda --pipeline 1 --size 128)
add_test(ThresholdTimingCuda2-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingCuda --pipeline 2 --size 128)
add_test(ThresholdTimingCuda3-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingCuda --pipeline 3 --size 128)

