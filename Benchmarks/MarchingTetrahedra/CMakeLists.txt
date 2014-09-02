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
##  Copyright 2014 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

#-----------------------------------------------------------------------------
macro(add_timing_tests target)
  add_test(${target}-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline=1 --size=128)
endmacro()

macro(add_resolveDuplicate_timing_tests target)
  add_test(${target}ResolveDuplicatePoints-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline=2 --size=128)
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
add_executable(MarchingTetrahedraTimingSerial ${sources} ${headers} )
set_dax_device_adapter(MarchingTetrahedraTimingSerial DAX_DEVICE_ADAPTER_SERIAL)
target_link_libraries(MarchingTetrahedraTimingSerial)
add_timing_tests(MarchingTetrahedraTimingSerial)
add_resolveDuplicate_timing_tests(MarchingTetrahedraTimingSerial)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(MarchingTetrahedraTimingOpenMP ${sources} ${headers})
  set_dax_device_adapter(MarchingTetrahedraTimingOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  target_link_libraries(MarchingTetrahedraTimingOpenMP)
  add_timing_tests(MarchingTetrahedraTimingOpenMP)
  add_resolveDuplicate_timing_tests(MarchingTetrahedraTimingOpenMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(MarchingTetrahedraTimingTBB ${sources} ${headers})
  set_dax_device_adapter(MarchingTetrahedraTimingTBB DAX_DEVICE_ADAPTER_TBB)
  target_link_libraries(MarchingTetrahedraTimingTBB ${TBB_LIBRARIES})
  add_timing_tests(MarchingTetrahedraTimingTBB)
  add_resolveDuplicate_timing_tests(MarchingTetrahedraTimingTBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  set(cuda_sources
    main.cu
    ArgumentsParser.cxx
    )

  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(MarchingTetrahedraTimingCuda ${cuda_sources} ${headers})
  set_dax_device_adapter(MarchingTetrahedraTimingCuda DAX_DEVICE_ADAPTER_CUDA)
  target_link_libraries(MarchingTetrahedraTimingCuda)
  add_timing_tests(MarchingTetrahedraTimingCuda)
  add_resolveDuplicate_timing_tests(MarchingTetrahedraTimingCuda)
endif (DAX_ENABLE_CUDA)
