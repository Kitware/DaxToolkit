#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
cmake_minimum_required(VERSION 2.8)
project (Dax)

OPTION(BUILD_CONTROL_ENVIRONMENT "Enable Control Environment" OFF)
OPTION(BUILD_EXECUTION_ENVIRONMENT "Enable Execution Environment" ON)

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

find_package(CUDA REQUIRED)

# List of Boost features used:
# * Smart Ptr
# * program options
find_package(Boost REQUIRED
  COMPONENTS program_options # program_options is only needed for the testing
                             # executables
  )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/daxOptions.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/daxOptions.h
  @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(Common)
add_subdirectory(DataModel)

if (BUILD_EXECUTION_ENVIRONMENT)
  # Add the execution environment
  add_subdirectory(ExecutionEnvironment)
endif()

if (BUILD_CONTROL_ENVIRONMENT)
  # add the control environment
  add_subdirectory(ControlEnvironment)
endif()
