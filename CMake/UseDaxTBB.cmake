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

if (Dax_TBB_initialize_complete)
  return()
endif (Dax_TBB_initialize_complete)

set(Dax_TBB_FOUND ${Dax_ENABLE_TBB})
if (NOT Dax_TBB_FOUND)
  message(STATUS "This build of Dax does not include TBB.")
endif (NOT Dax_TBB_FOUND)

# Find the Boost library.
if (Dax_TBB_FOUND)
  if(NOT Boost_FOUND)
    find_package(BoostHeaders ${Dax_REQUIRED_BOOST_VERSION})
  endif()

  if (NOT Boost_FOUND)
    message(STATUS "Boost not found")
    set(Dax_TBB_FOUND)
  endif (NOT Boost_FOUND)
endif (Dax_TBB_FOUND)

# Find TBB support.
if (Dax_TBB_FOUND)
  # FindTBB is not distributed with CMake, so find the one distributed
  # with Dax.
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${Dax_CMAKE_MODULE_PATH})

  find_package(TBB)

  if (NOT TBB_FOUND)
    message(STATUS "TBB not found")
    set(Dax_TBB_FOUND)
  endif (NOT TBB_FOUND)
endif (Dax_TBB_FOUND)

# Set up all these dependent packages (if they were all found).
if (Dax_TBB_FOUND)
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
    ${Dax_INCLUDE_DIRS}
    )
  link_libraries(${TBB_LIBRARIES})

  set(Dax_TBB_initialize_complete TRUE)
endif (Dax_TBB_FOUND)
