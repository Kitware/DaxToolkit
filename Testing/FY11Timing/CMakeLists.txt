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

#find_package(CUDA REQUIRED)

include_directories(${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

if (DAX_ENABLE_THRUST)
  include_directories(${THRUST_INCLUDE_DIR})
endif (DAX_ENABLE_THRUST)

#-----------------------------------------------------------------------------
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FY11TimingConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/FY11TimingConfig.h
  @ONLY
  )

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  PipelineDebug.cxx
  FY11Timing.cxx
  ArgumentsParser.cxx
  Timer.cxx
  )

if (DAX_ENABLE_OPENMP)
  set(sources ${sources} PipelineOpenMP.cxx)
endif (DAX_ENABLE_OPENMP)

add_executable(FY11Timing ${sources} ${headers})

set_source_files_properties(Pipeline.h PROPERTIES HEADER_FILE_ONLY TRUE)

target_link_libraries(FY11Timing
  ${Boost_LIBRARIES}
  )

#-----------------------------------------------------------------------------
add_test(FY11Timing1-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 1 --size 128)
add_test(FY11Timing2-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 2 --size 128)
add_test(FY11Timing3-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 3 --size 128)
#add_test(FY11Timing1-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 1 --size 256)
#add_test(FY11Timing2-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 2 --size 256)
#add_test(FY11Timing3-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 3 --size 256)
