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
set(headers
  BlackScholes.h
  )

#-----------------------------------------------------------------------------
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)

#-----------------------------------------------------------------------------
add_executable(BlackScholesSerial ${headers} main.cxx)
set_dax_device_adapter(BlackScholesSerial DAX_DEVICE_ADAPTER_SERIAL)
add_test(BlackScholesSerial ${EXECUTABLE_OUTPUT_PATH}/BlackScholesSerial)


#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(BlackScholesOpenMP ${headers} main.cxx)
  set_dax_device_adapter(BlackScholesOpenMP DAX_DEVICE_ADAPTER_OPENMP)
  add_test(BlackScholesOpenMP ${EXECUTABLE_OUTPUT_PATH}/BlackScholesOpenMP)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(BlackScholesTBB ${headers} main.cxx)
  set_dax_device_adapter(BlackScholesTBB DAX_DEVICE_ADAPTER_TBB)
  add_test(BlackScholesTBB ${EXECUTABLE_OUTPUT_PATH}/BlackScholesTBB)
  target_link_libraries(BlackScholesTBB ${TBB_LIBRARIES})
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(BlackScholesCuda ${headers} main.cu)
  set_dax_device_adapter(BlackScholesCuda DAX_DEVICE_ADAPTER_CUDA)
  add_test(BlackScholesCuda ${EXECUTABLE_OUTPUT_PATH}/BlackScholesCuda)

endif (DAX_ENABLE_CUDA)
