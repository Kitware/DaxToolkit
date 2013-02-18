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
set(headers
  BlackScholes.h
  )

#-----------------------------------------------------------------------------
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

#-----------------------------------------------------------------------------
add_executable(BlackScholesSerial ${headers} main.cxx)
add_test(BlackScholesSerial ${EXECUTABLE_OUTPUT_PATH}/BlackScholesSerial)
set_target_properties(BlackScholesSerial PROPERTIES COMPILE_FLAGS
                      -DDAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_SERIAL)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(BlackScholesOpenMP ${headers} main.cxx)
  set_target_properties(BlackScholesOpenMP PROPERTIES COMPILE_FLAGS
                        -DDAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_OPENMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(BlackScholesTBB ${headers} main.cxx)
  set_target_properties(BlackScholesTBB PROPERTIES COMPILE_FLAGS
                        -DDAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_TBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(BlackScholesCuda ${headers} main.cu)
  set_target_properties(BlackScholesCuda PROPERTIES COMPILE_FLAGS
                        -DDAX_DEVICE_ADAPTER=DAX_DEVICE_ADAPTER_CUDA)
  add_test(BlackScholesCuda
    ${EXECUTABLE_OUTPUT_PATH}/BlackScholesCuda)
endif (DAX_ENABLE_CUDA)
