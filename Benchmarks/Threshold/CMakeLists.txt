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
#Include VTK so we can benchmark agianst it
mark_as_advanced(DAX_ENABLE_VTK_THRESHOLD_BENCHMARK)
option(DAX_ENABLE_VTK_THRESHOLD_BENCHMARK FALSE)
if(DAX_ENABLE_VTK_THRESHOLD_BENCHMARK)
  find_package(VTK REQUIRED)
endif()

#-----------------------------------------------------------------------------
#Include PISTON so we can benchmark agianst it
mark_as_advanced(DAX_ENABLE_PISTON_THRESHOLD_BENCHMARK)
option(DAX_ENABLE_PISTON_THRESHOLD_BENCHMARK FALSE)
if(DAX_ENABLE_PISTON_THRESHOLD_BENCHMARK)
  find_package(VTK REQUIRED)
  find_path( PISTON_INCLUDE
    NAMES piston/piston_math.h
    DOC "Piston headers"
    )
endif()

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
  ArgumentsParser.cxx
  Timer.cxx
  )

#-----------------------------------------------------------------------------
if(DAX_ENABLE_VTK_THRESHOLD_BENCHMARK)
  set(vtkHeaders
    VTKPipeline.h
    )

  include(${VTK_USE_FILE})
  add_executable(ThresholdTimingVTK ${sources} ${vtkHeaders} PipelineVTK.cxx)
  set_source_files_properties(${vtkHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(ThresholdTimingVTK
    ${Boost_LIBRARIES}
    vtkFiltering
    vtkGraphics
    )

  add_test(ThresholdTimingVTK1-128
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingVTK --pipeline 1 --size 128)
  add_test(ThresholdTimingVTK1-256
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingVTK --pipeline 1 --size 256)
endif()

#-----------------------------------------------------------------------------
if(DAX_ENABLE_PISTON_THRESHOLD_BENCHMARK)
  set(pistonHeaders
    PISTONPipeline.h
    )
  include(${VTK_USE_FILE})
  include_directories(${PISTON_INCLUDE})
  cuda_add_executable(ThresholdTimingPISTON ${sources} ${pistonHeaders} PipelinePISTON.cu)
  set_source_files_properties(${pistonHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(ThresholdTimingPISTON
    ${Boost_LIBRARIES}
    vtkGraphics
    vtkFiltering
    )

  add_test(ThresholdTimingPiston1-128
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingPISTON --pipeline 1 --size 128)
  add_test(ThresholdTimingPiston1-256
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingPISTON --pipeline 1 --size 256)
endif()

#-----------------------------------------------------------------------------
add_executable(ThresholdTimingSerial ${sources} ${headers} PipelineSerial.cxx)
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_link_libraries(ThresholdTimingSerial
  ${Boost_LIBRARIES}
  )

add_test(ThresholdTimingSerial1-128
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingSerial --pipeline 1 --size 128)
add_test(ThresholdTimingSerial1-256
  ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingSerial --pipeline 1 --size 256)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(ThresholdTimingOpenMP ${sources} ${headers} PipelineOpenMP.cxx)
  target_link_libraries(ThresholdTimingOpenMP
    ${Boost_LIBRARIES}
    )

  add_test(ThresholdTimingOpenMP1-128
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingOpenMP --pipeline 1 --size 128)
  add_test(ThresholdTimingOpenMP1-256
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingOpenMP --pipeline 1 --size 256)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  cuda_add_executable(ThresholdTimingCuda
    ${sources} ${headers} PipelineCuda.h PipelineCuda.cu
    )
  set_source_files_properties(PipelineCuda.h PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(ThresholdTimingCuda
    ${Boost_LIBRARIES}
    )

  add_test(ThresholdTimingCuda1-128
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingCuda --pipeline 1 --size 128)
  add_test(ThresholdTimingCuda1-256
    ${EXECUTABLE_OUTPUT_PATH}/ThresholdTimingCuda --pipeline 1 --size 256)
endif (DAX_ENABLE_CUDA)

