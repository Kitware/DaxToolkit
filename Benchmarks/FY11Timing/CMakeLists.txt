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
  add_test(${target}1-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline=1 --size=128)
  add_test(${target}2-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline=2 --size=128)
  add_test(${target}3-128
    ${EXECUTABLE_OUTPUT_PATH}/${target} --pipeline=3 --size=128)
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
add_executable(FY11TimingSerial ${sources} ${headers} )
set_dax_device_adapter(FY11TimingSerial DAX_DEVICE_ADAPTER_SERIAL)
target_link_libraries(FY11TimingSerial)
add_timing_tests(FY11TimingSerial)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(FY11TimingOpenMP ${sources} ${headers})
  set_dax_device_adapter(FY11TimingOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  target_link_libraries(FY11TimingOpenMP)
  add_timing_tests(FY11TimingOpenMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(FY11TimingTBB ${sources} ${headers})
  set_dax_device_adapter(FY11TimingTBB DAX_DEVICE_ADAPTER_TBB)
  target_link_libraries(FY11TimingTBB ${DAX_TIMING_LIBS})
  add_timing_tests(FY11TimingTBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  set(cuda_sources
    main.cu
    ArgumentsParser.cxx
    )

  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(FY11TimingCuda ${cuda_sources} ${headers})
  set_dax_device_adapter(FY11TimingCuda DAX_DEVICE_ADAPTER_CUDA)
  target_link_libraries(FY11TimingCuda)
  add_timing_tests(FY11TimingCuda)
endif (DAX_ENABLE_CUDA)

