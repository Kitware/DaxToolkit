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

# Find the Boost library.
if (Dax_Serial_FOUND)
  find_package(Boost ${Dax_REQUIRED_BOOST_VERSION})

  if (NOT Boost_FOUND)
    message(STATUS "Boost not found")
    set(Dax_Serial_FOUND)
  endif (NOT Boost_FOUND)
endif (Dax_Serial_FOUND)

# Set up all these dependent packages (if they were all found).
if (Dax_Serial_FOUND)
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${Dax_INCLUDE_DIRS}
    )
endif (Dax_Serial_FOUND)
