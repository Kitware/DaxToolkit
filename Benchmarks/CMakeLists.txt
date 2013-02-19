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
#verify that all devices are setup properly
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
# create a macro to easily set the device adapter on a target
macro(set_dax_device_adapter target adapter)
  set_target_properties(${target} PROPERTIES COMPILE_FLAGS
                        -DDAX_DEVICE_ADAPTER=${adapter})
endmacro()

#-----------------------------------------------------------------------------
add_subdirectory(BlackScholes)
add_subdirectory(FY11Timing)
add_subdirectory(MarchingCubes)
add_subdirectory(Threshold)
