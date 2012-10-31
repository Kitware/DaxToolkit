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

if (Dax_OpenMP_initialize_complete)
  return()
endif (Dax_OpenMP_initialize_complete)

set(Dax_OpenMP_FOUND ${Dax_ENABLE_OPENMP})
if (NOT Dax_OpenMP_FOUND)
  message(STATUS "This build of Dax does not include OpenMP.")
endif (NOT Dax_OpenMP_FOUND)

# Find the Boost library.
if (Dax_OpenMP_FOUND)
  find_package(Boost ${Dax_REQUIRED_BOOST_VERSION})

  if (NOT Boost_FOUND)
    message(STATUS "Boost not found")
    set(Dax_OpenMP_FOUND)
  endif (NOT Boost_FOUND)
endif (Dax_OpenMP_FOUND)

# Find the Thrust library.
if (Dax_OpenMP_FOUND)
  # FindThrust is not distributed with CMake, so find the one distributed
  # with Dax.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${Dax_CMAKE_MODULE_PATH})

  find_package(Thrust)

  if (NOT THRUST_FOUND)
    message(STATUS "Thrust not found")
    set(Dax_OpenMP_FOUND)
  endif (NOT THRUST_FOUND)
endif (Dax_OpenMP_FOUND)

# Find OpenMP support.
if (Dax_OpenMP_FOUND)
  find_package(OpenMP)

  if (NOT OPENMP_FOUND)
    message(STATUS "OpenMP not found")
    set(Dax_OpenMP_FOUND)
  endif (NOT OPENMP_FOUND)
endif (Dax_OpenMP_FOUND)

# Set up all these dependent packages (if they were all found).
if (Dax_OpenMP_FOUND)
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${THRUST_INCLUDE_DIR}
    ${Dax_INCLUDE_DIRS}
    )

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

  set(Dax_OpenMP_initialize_complete TRUE)
endif (Dax_OpenMP_FOUND)
