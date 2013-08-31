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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGXX)

  include(CheckCXXCompilerFlag)

  # Standard warning flags we should always have
  set(CMAKE_CXX_FLAGS_WARN " -Wall")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_WARN}")
  set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_WARN}")

  # Addtional warnings for GCC
  set(CMAKE_CXX_FLAGS_WARN_EXTRA "-Wno-long-long -Wcast-align -Wchar-subscripts -Wextra -Wpointer-arith -Wformat -Wformat-security -Wshadow -Wunused-parameter -fno-common")
  if (NOT DAX_FORCE_ANSI AND HAS_CXX11_VARIADIC_TEMPLATES)
    set(CMAKE_CXX_FLAGS_WARN_EXTRA "-std=c++11 ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
  else (NOT DAX_FORCE_ANSI AND HAS_CXX11_VARIADIC_TEMPLATES)
    set(CMAKE_CXX_FLAGS_WARN_EXTRA "-ansi ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
  endif (NOT DAX_FORCE_ANSI AND HAS_CXX11_VARIADIC_TEMPLATES)

  # Set up the debug CXX_FLAGS for extra warnings
  option(DAX_EXTRA_COMPILER_WARNINGS "Add compiler flags to do stricter checking when building debug." ON)
  # We used to add the compiler flags globally, but this caused problems with
  # the CUDA compiler (and its lack of support for GCC pragmas).  Instead,
  # the dax_declare_headers and dax_unit_tests CMake functions add these flags
  # to their compiles.  As long as the unit tests have good coverage, this
  # should catch all problems.
  if(DAX_EXTRA_COMPILER_WARNINGS)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_WARN_EXTRA}")
  endif()

  #add in support for debugging Thrust when building in debug mode
  set(CMAKE_CXX_FLAGS_DEBUG_THRUST "-DTHRUST_DEBUG")
  option(DAX_DEBUG_THRUST "Add in support for thrust debugging" OFF)
  if(DAX_DEBUG_THRUST)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_DEBUG_THRUST}")
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG_THRUST}")
  endif()

endif()

