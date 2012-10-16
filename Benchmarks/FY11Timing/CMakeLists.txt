##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

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
  ${CMAKE_CURRENT_SOURCE_DIR}/FY11TimingConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/FY11TimingConfig.h
  @ONLY
  )

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  #PipelineDebug.cxx
  #FY11Timing.cxx
  ArgumentsParser.cxx
  Timer.cxx
  )

#-----------------------------------------------------------------------------
add_executable(FY11TimingSerial ${sources} ${headers} PipelineSerial.cxx)
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_link_libraries(FY11TimingSerial
  ${Boost_LIBRARIES}
  )

add_test(FY11TimingSerial1-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 1 --size 128)
add_test(FY11TimingSerial2-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 2 --size 128)
add_test(FY11TimingSerial3-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 3 --size 128)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(FY11TimingOpenMP ${sources} ${headers} PipelineOpenMP.cxx)
  target_link_libraries(FY11TimingOpenMP
    ${Boost_LIBRARIES}
    )

  add_test(FY11TimingOpenMP1-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 1 --size 128)
  add_test(FY11TimingOpenMP2-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 2 --size 128)
  add_test(FY11TimingOpenMP3-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 3 --size 128)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(FY11TimingCuda
    ${sources} ${headers} PipelineCuda.h PipelineCuda.cu
    )
  set_source_files_properties(PipelineCuda.h PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(FY11TimingCuda
    ${Boost_LIBRARIES}
    )

  add_test(FY11TimingCuda1-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 1 --size 128)
  add_test(FY11TimingCuda2-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 2 --size 128)
  add_test(FY11TimingCuda3-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 3 --size 128)
endif (DAX_ENABLE_CUDA)


#-----------------------------------------------------------------------------
#add_test(FY11Timing1-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 1 --size 256)
#add_test(FY11Timing2-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 2 --size 256)
#add_test(FY11Timing3-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 3 --size 256)
