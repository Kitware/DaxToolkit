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

if (Dax_Serial_initialize_complete)
  return()
endif (Dax_Serial_initialize_complete)

# Find the Boost library.
if (NOT Dax_Serial_FOUND)
  if(NOT Boost_FOUND)
    find_package(BoostHeaders ${Dax_REQUIRED_BOOST_VERSION})
  endif()

  if (NOT Boost_FOUND)
    message(STATUS "Boost not found")
    set(Dax_Serial_FOUND FALSE)
  else(NOT Boost_FOUND)
    set(Dax_Serial_FOUND TRUE)
  endif (NOT Boost_FOUND)
endif (NOT Dax_Serial_FOUND)

# Set up all these dependent packages (if they were all found).
if (Dax_Serial_FOUND)
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${Dax_INCLUDE_DIRS}
    )

  set(Dax_Serial_initialize_complete TRUE)
endif (Dax_Serial_FOUND)
