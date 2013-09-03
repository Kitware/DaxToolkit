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

if (Dax_Cuda_initialize_complete)
  return()
endif (Dax_Cuda_initialize_complete)

set(Dax_Cuda_FOUND ${Dax_ENABLE_CUDA})
if (NOT Dax_Cuda_FOUND)
  message(STATUS "This build of Dax does not include Cuda.")
endif (NOT Dax_Cuda_FOUND)

# Find the Boost library.
if (Dax_Cuda_FOUND)
  if(NOT Boost_FOUND)
    find_package(BoostHeaders ${Dax_REQUIRED_BOOST_VERSION})
  endif()

  if (NOT Boost_FOUND)
    message(STATUS "Boost not found")
    set(Dax_Cuda_FOUND)
  endif (NOT Boost_FOUND)
endif (Dax_Cuda_FOUND)

# Find the Thrust library.
if (Dax_Cuda_FOUND)
  find_package(Thrust)

  if (NOT THRUST_FOUND)
    message(STATUS "Thrust not found")
    set(Dax_Cuda_FOUND)
  endif (NOT THRUST_FOUND)
endif (Dax_Cuda_FOUND)

# Find Cuda support.
if (Dax_Cuda_FOUND)
  find_package(CUDA)

  mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)

  if (NOT CUDA_FOUND)
    message(STATUS "CUDA not found")
    set(Dax_Cuda_FOUND)
  endif (NOT CUDA_FOUND)
endif (Dax_Cuda_FOUND)

# Set up all these dependent packages (if they were all found).
if (Dax_Cuda_FOUND)
  cuda_include_directories(
    ${Boost_INCLUDE_DIRS}
    ${THRUST_INCLUDE_DIR}
    ${Dax_INCLUDE_DIRS}
    )
  set(Dax_Cuda_initialize_complete TRUE)
endif (Dax_Cuda_FOUND)
