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
##  Copyright 2011 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

cmake_minimum_required(VERSION 2.8)
project (Dax)

set(Dax_MAJOR_VERSION 0)
set(Dax_MINOR_VERSION 1)
set(Dax_PATCH_VERSION 0)
set(Dax_VERSION "${Dax_MAJOR_VERSION}.${Dax_MINOR_VERSION}.${Dax_PATCH_VERSION}")

set(Dax_INSTALL_INCLUDE_DIR "include")
set(Dax_INSTALL_CONFIG_DIR "include")
set(Dax_INSTALL_CMAKE_MODULE_DIR "share/dax/cmake")

set(Dax_REQUIRED_BOOST_VERSION 1.48.0)
set(Dax_REQUIRED_THRUST_VERSION 1.4.0)

# include some dax-specific cmake code.
include(CMake/DaxMacros.cmake)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${Dax_SOURCE_DIR}/CMake)

#-----------------------------------------------------------------------------
# Add supplemental compiler warnings, and GCC visibility support.
include(CMake/DaxCompilerExtras.cmake)

#-----------------------------------------------------------------------------
# Configurable Options
option(DAX_ENABLE_CUDA "Enable Cuda support" ON)
option(DAX_ENABLE_OPENMP "Enable OpenMP support" ON)
option(DAX_ENABLE_TBB "Enable TBB support" OFF)
option(DAX_ENABLE_TESTING "Enable DAX Testing" ON)
option(DAX_ENABLE_DOXYGEN
  "Enable DAX Documentation Generation (Needs Doxygen)" OFF)
option(DAX_DISABLE_KITS
  "Internal option useful on machines generating documentation alone"
  OFF)
mark_as_advanced(DAX_DISABLE_KITS)

option(DAX_USE_DOUBLE_PRECISION
  "Use double precision for floating point calculations"
  OFF
  )
option(DAX_USE_64BIT_IDS "Use 64-bit indices." OFF)

if (DAX_ENABLE_CUDA OR DAX_ENABLE_OPENMP)
  set(DAX_ENABLE_THRUST ON)
endif (DAX_ENABLE_CUDA OR DAX_ENABLE_OPENMP)

if (DAX_ENABLE_TESTING)
  enable_testing()
  include(CTest)
endif()

if (DAX_ENABLE_DOXYGEN)
  add_subdirectory(Doxygen)
endif()

#-----------------------------------------------------------------------------
# setup timing libs we need to link too
set(DAX_TIMING_LIBS)
if(UNIX AND NOT APPLE)
  set(DAX_TIMING_LIBS rt)
endif()

#-----------------------------------------------------------------------------
# Set up devices selected.
dax_configure_device(Serial)
if (DAX_ENABLE_CUDA)
  dax_configure_device(Cuda)
endif (DAX_ENABLE_CUDA)
if (DAX_ENABLE_OPENMP)
  dax_configure_device(OpenMP)
endif (DAX_ENABLE_OPENMP)
if (DAX_ENABLE_TBB)
  dax_configure_device(TBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------

## Set the directory where the binaries will be stored
set( EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executable will be stored"
)

## Set the directory where the libraries will be stored
set( LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/libs
  CACHE PATH
  "Directory where all the libraries will be stored"
)
mark_as_advanced(
  EXECUTABLE_OUTPUT_PATH
  LIBRARY_OUTPUT_PATH)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

#-----------------------------------------------------------------------------
# Add test to check for copyright statement on all source files.
if (DAX_ENABLE_TESTING)
  add_test(NAME CopyrightStatement
    COMMAND ${CMAKE_COMMAND} "-DDax_SOURCE_DIR=${Dax_SOURCE_DIR}" -P "${Dax_SOURCE_DIR}/CMake/DaxCheckCopyright.cmake"
    )
endif (DAX_ENABLE_TESTING)

#-----------------------------------------------------------------------------
# Check basic type sizes.
include(CheckTypeSize)

check_type_size(float DAX_SIZE_FLOAT BUILTIN_TYPES_ONLY)
check_type_size(double DAX_SIZE_DOUBLE BUILTIN_TYPES_ONLY)
check_type_size(int DAX_SIZE_INT BUILTIN_TYPES_ONLY)
check_type_size(long DAX_SIZE_LONG BUILTIN_TYPES_ONLY)
check_type_size("long long" DAX_SIZE_LONG_LONG BUILTIN_TYPES_ONLY)

#-----------------------------------------------------------------------------
# Build the configure file.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dax/internal/Configure.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/dax/internal/Configure.h
  @ONLY)
dax_install_headers(
  dax/internal ${CMAKE_CURRENT_BINARY_DIR}/dax/internal/Configure.h)


if (NOT DAX_DISABLE_KITS)

  # List of Boost features used:
  # * Smart Ptr
  # * Meta programming language
  find_package(Boost ${Dax_REQUIRED_BOOST_VERSION} REQUIRED)

#-----------------------------------------------------------------------------
# Add subdirectories
add_subdirectory(dax)

if (DAX_ENABLE_TESTING)
  add_subdirectory(Benchmarks)
endif (DAX_ENABLE_TESTING)

endif(NOT DAX_DISABLE_KITS)

# Configuration for build directory.
set(Dax_INCLUDE_DIRS_CONFIG "${Dax_SOURCE_DIR};${Dax_BINARY_DIR}")
set(Dax_CMAKE_MODULE_PATH_CONFIG "${Dax_SOURCE_DIR}/CMake")
configure_file(
  ${Dax_SOURCE_DIR}/CMake/DaxConfig.cmake.in
  ${Dax_BINARY_DIR}/DaxConfig.cmake
  @ONLY IMMEDIATE
  )

# Configuration for install.
set(Dax_INCLUDE_DIRS_CONFIG "\${_install_dir}/${Dax_INSTALL_INCLUDE_DIR}")
set(Dax_CMAKE_MODULE_PATH_CONFIG "\${_install_dir}/${Dax_INSTALL_CMAKE_MODULE_DIR}")
configure_file(
  ${Dax_SOURCE_DIR}/CMake/DaxConfig.cmake.in
  ${Dax_BINARY_DIR}/DaxConfig.cmake.install
  @ONLY IMMEDIATE
  )
install(FILES ${Dax_BINARY_DIR}/DaxConfig.cmake.install
  DESTINATION ${Dax_INSTALL_CONFIG_DIR}
  RENAME DaxConfig.cmake
  )

# Create supplemental version configuration file.
configure_file(
  ${Dax_SOURCE_DIR}/CMake/DaxConfigVersion.cmake.in
  ${Dax_BINARY_DIR}/DaxConfigVersion.cmake
  @ONLY
  )
install(FILES ${Dax_BINARY_DIR}/DaxConfigVersion.cmake
  DESTINATION ${Dax_INSTALL_CONFIG_DIR}
  )

# Install the readme and license files.
install(FILES ${Dax_SOURCE_DIR}/README.txt
  DESTINATION ${Dax_INSTALL_CONFIG_DIR}
  RENAME DaxREADME.txt
  )
install(FILES ${Dax_SOURCE_DIR}/LICENSE.txt
  DESTINATION ${Dax_INSTALL_CONFIG_DIR}
  RENAME DaxLICENSE.txt
  )

# Install helper configure files.
install(
  FILES
    ${Dax_SOURCE_DIR}/CMake/FindTBB.cmake
    ${Dax_SOURCE_DIR}/CMake/FindThrust.cmake
  DESTINATION ${Dax_INSTALL_CMAKE_MODULE_DIR}
  )
install(
  FILES
    ${Dax_SOURCE_DIR}/CMake/UseDaxSerial.cmake
    ${Dax_SOURCE_DIR}/CMake/UseDaxOpenMP.cmake
    ${Dax_SOURCE_DIR}/CMake/UseDaxCuda.cmake
    ${Dax_SOURCE_DIR}/CMake/UseDaxTBB.cmake
  DESTINATION ${Dax_INSTALL_CMAKE_MODULE_DIR}
  )

# Enable CPack packaging
set(CPACK_PACKAGE_DESCRIPTION_FILE ${Dax_SOURCE_DIR}/README.txt)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Dax Toolkit")
set(CPACK_PACKAGE_NAME "Dax")
set(CPACK_PACKAGE_VENDOR "Sandia National Laboratories, Kitware Inc., and UC Davis")
set(CPACK_PACKAGE_VERSION_MAJOR ${Dax_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${Dax_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${Dax_PATCH_VERSION})
set(CPACK_PACKAGE_FILE_NAME "Dax-${Dax_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE ${Dax_SOURCE_DIR}/LICENSE.txt)
set(CPACK_RESOURCE_FILE_README ${Dax_SOURCE_DIR}/README.txt)
include(CPack)
