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
add_executable(ThresholdTimingSerial ${sources} ${headers} )
set_dax_device_adapter(ThresholdTimingSerial DAX_DEVICE_ADAPTER_SERIAL)
target_link_libraries(ThresholdTimingSerial ${Boost_LIBRARIES})
add_timing_tests(ThresholdTimingSerial)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(ThresholdTimingOpenMP ${sources} ${headers})
  set_dax_device_adapter(ThresholdTimingOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  target_link_libraries(ThresholdTimingOpenMP ${Boost_LIBRARIES})
  add_timing_tests(ThresholdTimingOpenMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(ThresholdTimingTBB ${sources} ${headers})
  set_dax_device_adapter(ThresholdTimingTBB DAX_DEVICE_ADAPTER_TBB)
  target_link_libraries(ThresholdTimingTBB ${Boost_LIBRARIES})
  add_timing_tests(ThresholdTimingTBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  set(cuda_sources
    main.cu
    ArgumentsParser.cxx
    )

  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(ThresholdTimingCuda ${cuda_sources} ${headers})
  set_dax_device_adapter(ThresholdTimingCuda DAX_DEVICE_ADAPTER_CUDA)
  target_link_libraries(ThresholdTimingCuda ${Boost_LIBRARIES})
  add_timing_tests(ThresholdTimingCuda)
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
  add_executable(ThresholdTimingVTK ${vtkHeaders} ${vtkSources})
  target_link_libraries(ThresholdTimingVTK
    ${Boost_LIBRARIES}
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkFiltersCore
    vtkFiltersGeometry
    )
  add_timing_tests(ThresholdTimingVTK)
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

    add_executable(ThresholdTimingOpenMPPiston
                    ${pistonHeaders} ${pistonSources})
    set_dax_device_adapter(ThresholdTimingOpenMPPiston
                            DAX_DEVICE_ADAPTER_OPENMP)
    target_link_libraries(ThresholdTimingOpenMPPiston ${Boost_LIBRARIES})
    add_timing_tests(ThresholdTimingOpenMPP)
  endif()

  if (DAX_ENABLE_CUDA)
    set(pistonSources
      mainPiston.cu
      ArgumentsParser.cxx
      )

    cuda_add_executable(ThresholdTimingCudaPiston
                          ${pistonHeaders} ${pistonSources})
    set_dax_device_adapter(ThresholdTimingCudaPiston
                            DAX_DEVICE_ADAPTER_CUDA)
    target_link_libraries(ThresholdTimingCudaPiston ${Boost_LIBRARIES})
    add_timing_tests(ThresholdTimingCudaPiston)
  endif()
endif()

