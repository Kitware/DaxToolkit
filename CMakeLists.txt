#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
cmake_minimum_required(VERSION 2.8)
project (Dax)

# include some dax-specific cmake code.
include(CMake/DaxMacros.cmake)

#-----------------------------------------------------------------------------
# Add supplemental compiler warnings, and GCC visibility support.
include(CMake/DaxCompilerExtras.cmake)

#-----------------------------------------------------------------------------
# Configurable Options
option(DAX_ENABLE_CUDA "Enable Cuda support" ON)
option(DAX_ENABLE_TESTING "Enable DAX Testing" ON)
option(DAX_ENABLE_DOXYGEN
  "Enable DAX Documentation Generation (Needs Doxygen)" OFF)
option(DAX_DISABLE_KITS
  "Internal option useful on machines generating documentation alone"
  OFF)
mark_as_advanced(DAX_DISABLE_KITS)

if (DAX_ENABLE_TESTING)
  enable_testing()
  include(CTest)
endif()

if (DAX_ENABLE_DOXYGEN)
  add_subdirectory(Doxygen)
endif()

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
# Configure the options file. There should be nothing more in this file besides
# #defines based on the CMake settings, if any.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/daxOptions.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/daxOptions.h
  @ONLY)


if (NOT DAX_DISABLE_KITS)

  # List of Boost features used:
  # * Smart Ptr
  # * program options
  # * timer
  find_package(Boost REQUIRED
    COMPONENTS
      program_options # only needed for the testing executables
      timer # only needed for the testing executables
      system # only need for timer for testing executables
  )

#-----------------------------------------------------------------------------
# Add subdirectories
add_subdirectory(dax)

if (DAX_ENABLE_TESTING)
  add_subdirectory(Testing)
endif (DAX_ENABLE_TESTING)

endif()
