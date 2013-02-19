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
macro(add_timing_tests target)
  add_test(${target}-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline 1 --size 128)
    add_test(${target}-256
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline 1 --size 256)
endmacro()

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  main.cxx
  ArgumentsParser.cxx
  )

set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

#-----------------------------------------------------------------------------
add_executable(MarchingCubesTimingSerial ${sources} ${headers} )
set_dax_device_adapter(MarchingCubesTimingSerial DAX_DEVICE_ADAPTER_SERIAL)
target_link_libraries(MarchingCubesTimingSerial ${Boost_LIBRARIES})
add_timing_tests(MarchingCubesTimingSerial)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(MarchingCubesTimingOpenMP ${sources} ${headers})
  set_dax_device_adapter(MarchingCubesTimingOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  target_link_libraries(MarchingCubesTimingOpenMP ${Boost_LIBRARIES})
  add_timing_tests(MarchingCubesTimingOpenMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(MarchingCubesTimingTBB ${sources} ${headers})
  set_dax_device_adapter(MarchingCubesTimingTBB DAX_DEVICE_ADAPTER_TBB)
  target_link_libraries(MarchingCubesTimingTBB ${Boost_LIBRARIES})
  add_timing_tests(MarchingCubesTimingTBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  set(cuda_sources
    main.cu
    ArgumentsParser.cxx
    )

  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(MarchingCubesTimingCuda ${cuda_sources} ${headers})
  set_dax_device_adapter(MarchingCubesTimingCuda DAX_DEVICE_ADAPTER_CUDA)
  target_link_libraries(MarchingCubesTimingCuda ${Boost_LIBRARIES})
  add_timing_tests(MarchingCubesTimingCuda)
endif (DAX_ENABLE_CUDA)



#-----------------------------------------------------------------------------
if(DAX_ENABLE_VTK_BENCHMARKS)
  set(vtkHeaders
      VTKPipeline.h
      )
  set_source_files_properties(${vtkHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)

  set(vtkSources
    mainVTK.cxx
    ArgumentsParser.cxx
    )

  include(${VTK_USE_FILE})
  add_executable(MarchingCubesTimingVTK ${vtkHeaders} ${vtkSources})
  target_link_libraries(MarchingCubesTimingVTK
    ${Boost_LIBRARIES}
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkCommonMisc
    vtkFiltersCore
    vtkFiltersGeometry
    )
  add_timing_tests(MarchingCubesTimingVTK)
endif()

#-----------------------------------------------------------------------------
if(DAX_ENABLE_PISTON_BENCHMARKS)

  include_directories(${PISTON_INCLUDE})

  set(pistonHeaders
    PistonPipeline.h
    )
  set_source_files_properties(${pistonHeaders} PROPERTIES HEADER_FILE_ONLY TRUE)

  if (DAX_ENABLE_OPENMP)
    set(pistonSources
      mainPiston.cxx
      ArgumentsParser.cxx
      )

    add_executable(MarchingCubesTimingOpenMPPiston
                    ${pistonHeaders} ${pistonSources})
    set_dax_device_adapter(MarchingCubesTimingOpenMPPiston
                            DAX_DEVICE_ADAPTER_OPENMP)
    target_link_libraries(MarchingCubesTimingOpenMPPiston ${Boost_LIBRARIES})
    add_timing_tests(MarchingCubesTimingOpenMPP)
  endif()

  if (DAX_ENABLE_CUDA)
    set(pistonSources
      mainPiston.cu
      ArgumentsParser.cxx
      )

    cuda_add_executable(MarchingCubesTimingCudaPiston
                          ${pistonHeaders} ${pistonSources})
    set_dax_device_adapter(MarchingCubesTimingCudaPiston
                            DAX_DEVICE_ADAPTER_CUDA)
    target_link_libraries(MarchingCubesTimingCudaPiston ${Boost_LIBRARIES})
    add_timing_tests(MarchingCubesTimingCudaPiston)
  endif()
endif()
