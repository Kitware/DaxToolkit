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
  dax_configure_device(OpenMP)
endif (DAX_ENABLE_OPENMP)

if (DAX_ENABLE_CUDA)
  dax_configure_device(Cuda)
endif (DAX_ENABLE_CUDA)

if (DAX_ENABLE_TBB)
  dax_configure_device(TBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
#Include VTK so we can benchmark agianst it
mark_as_advanced(DAX_ENABLE_VTK_MARCHING_CUBES_BENCHMARK)
option(DAX_ENABLE_VTK_MARCHING_CUBES_BENCHMARK FALSE)
if(DAX_ENABLE_VTK_MARCHING_CUBES_BENCHMARK)
  find_package(VTK 6.0
    COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkCommonMisc
    vtkFiltersCore
    vtkFiltersGeometry
    REQUIRED)
endif()

#-----------------------------------------------------------------------------
#Include PISTON so we can benchmark agianst it
mark_as_advanced(DAX_ENABLE_PISTON_MARCHING_CUBES_BENCHMARK)
option(DAX_ENABLE_PISTON_MARCHING_CUBES_BENCHMARK FALSE)
if(DAX_ENABLE_PISTON_MARCHING_CUBES_BENCHMARK)
  find_package(VTK 6.0
    COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkCommonMisc
    vtkFiltersCore
    vtkFiltersGeometry
    REQUIRED)

  find_path( PISTON_INCLUDE
    NAMES piston/piston_math.h
    DOC "Piston headers"
    )
endif()

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  ArgumentsParser.cxx
  )

#-----------------------------------------------------------------------------
if(DAX_ENABLE_VTK_MARCHING_CUBES_BENCHMARK)
  set(vtkHeaders
    VTKPipeline.h
    )

  include(${VTK_USE_FILE})
  add_executable(MarchingCubesTimingVTK ${sources} ${vtkHeaders} PipelineVTK.cxx)
  set_source_files_properties(${vtkHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(MarchingCubesTimingVTK
    ${Boost_LIBRARIES}
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkCommonMisc
    vtkFiltersCore
    vtkFiltersGeometry
    )

  add_test(MarchingCubesTimingVTK1-128
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingVTK --pipeline 1 --size 128)
  add_test(MarchingCubesTimingVTK1-256
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingVTK --pipeline 1 --size 256)
endif()

#-----------------------------------------------------------------------------
if(DAX_ENABLE_PISTON_MARCHING_CUBES_BENCHMARK)
  set(pistonHeaders
    PistonPipeline.h
    )
  include(${VTK_USE_FILE})
  include_directories(${PISTON_INCLUDE})

  if (DAX_ENABLE_OPENMP)
    add_executable(MarchingCubesTimingOpenMPPiston ${sources} ${pistonHeaders} PipelineOpenMPPiston.cxx)
    target_link_libraries(MarchingCubesTimingOpenMPPiston
      ${Boost_LIBRARIES}
      vtkCommonCore
      vtkCommonDataModel
      vtkCommonExecutionModel
      vtkCommonMisc
      vtkFiltersCore
      vtkFiltersGeometry
      )

    add_test(MarchingCubesTimingOpenMPP1-128
      ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingOpenMPPiston --pipeline 1 --size 128)
    add_test(MarchingCubesTimingOpenMP1-256
      ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingOpenMPPiston --pipeline 1 --size 256)
  endif()

  if (DAX_ENABLE_CUDA)
    cuda_add_executable(MarchingCubesTimingCudaPiston ${sources} ${pistonHeaders} PipelineCudaPiston.cu)
    set_source_files_properties(${pistonHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)
    target_link_libraries(MarchingCubesTimingCudaPiston
      ${Boost_LIBRARIES}
      vtkCommonCore
      vtkCommonDataModel
      vtkCommonExecutionModel
      vtkCommonMisc
      vtkFiltersCore
      vtkFiltersGeometry
      )

    add_test(MarchingCubesTimingPiston1-128
      ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingCudaPiston --pipeline 1 --size 128)
    add_test(MarchingCubesTimingPiston1-256
      ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingCudaPiston --pipeline 1 --size 256)
  endif()
endif()

#-----------------------------------------------------------------------------
add_executable(MarchingCubesTimingSerial ${sources} ${headers} PipelineSerial.cxx)
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_link_libraries(MarchingCubesTimingSerial
  ${Boost_LIBRARIES}
  )

add_test(MarchingCubesTimingSerial1-128
  ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingSerial --pipeline 1 --size 128)
add_test(MarchingCubesTimingSerial1-256
  ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingSerial --pipeline 1 --size 256)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(MarchingCubesTimingOpenMP ${sources} ${headers} PipelineOpenMP.cxx)
  target_link_libraries(MarchingCubesTimingOpenMP
    ${Boost_LIBRARIES}
    )

  add_test(MarchingCubesTimingOpenMP1-128
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingOpenMP --pipeline 1 --size 128)
  add_test(MarchingCubesTimingOpenMP1-256
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingOpenMP --pipeline 1 --size 256)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(MarchingCubesTimingTBB ${sources} ${headers} PipelineTBB.cxx)
  target_link_libraries(MarchingCubesTimingTBB
    ${Boost_LIBRARIES}
    )

  add_test(MarchingCubesTimingTBB1-128
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingTBB --pipeline 1 --size 128)
  add_test(MarchingCubesTimingTBB1-256
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingTBB --pipeline 1 --size 256)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(MarchingCubesTimingCuda
    ${sources} ${headers} PipelineCuda.h PipelineCuda.cu
    )
  set_source_files_properties(PipelineCuda.h PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(MarchingCubesTimingCuda
    ${Boost_LIBRARIES}
    )

  add_test(MarchingCubesTimingCuda1-128
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingCuda --pipeline 1 --size 128)
  add_test(MarchingCubesTimingCuda1-256
    ${EXECUTABLE_OUTPUT_PATH}/MarchingCubesTimingCuda --pipeline 1 --size 256)
endif (DAX_ENABLE_CUDA)

