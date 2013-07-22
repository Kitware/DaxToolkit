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
##  Copyright 2013 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================
include(CMakeForceCompiler)

# this one is important
set(CMAKE_SYSTEM_NAME Linux)

# this will allow you to add even more xeon phi options
# by creating the file under your cmake module directory
# Handy if you want to play with performance tuning
# Platform/Linux-icpc-XeonPhi.cmake
set(CMAKE_SYSTEM_PROCESSOR XeonPhi)

# specify the cross compiler
CMAKE_FORCE_C_COMPILER(icc Intel)
CMAKE_FORCE_CXX_COMPILER(icpc Intel)

# where is the target environment
# maybe: intel64-location/lib/mic/ in your case
set(CMAKE_FIND_ROOT_PATH  /opt/intel/composer_xe_2013.2.146/compiler/lib/mic/ )

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

#make sure the first argument to the intel compilers is the mmic option to
#enable compilation for the Xeon Phi.
set(CMAKE_CXX_COMPILER_ARG1 -mmic)
set(CMAKE_C_COMPILER_ARG1 -mmic)
