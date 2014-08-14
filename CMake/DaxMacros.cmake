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

include(CMakeParseArguments)

# Utility to build a kit name from the current directory.
function(dax_get_kit_name kitvar)
  # Will this always work?  It should if ${CMAKE_CURRENT_SOURCE_DIR} is
  # built from ${Dax_SOURCE_DIR}.
  string(REPLACE "${Dax_SOURCE_DIR}/" "" dir_prefix ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" kit "${dir_prefix}")
  set(${kitvar} "${kit}" PARENT_SCOPE)
  # Optional second argument to get dir_prefix.
  if (${ARGC} GREATER 1)
    set(${ARGV1} "${dir_prefix}" PARENT_SCOPE)
  endif (${ARGC} GREATER 1)
endfunction(dax_get_kit_name)

# Builds a source file and an executable that does nothing other than
# compile the given header files.
function(dax_add_header_build_test name dir_prefix use_cuda)
  set(hfiles ${ARGN})
  if (use_cuda)
    set(suffix ".cu")
  else (use_cuda)
    set(suffix ".cxx")
  endif (use_cuda)
  set(cxxfiles)
  foreach (header ${ARGN})
    get_source_file_property(cant_be_tested ${header} DAX_CANT_BE_HEADER_TESTED)

    if( NOT cant_be_tested )
      string(REPLACE "${CMAKE_CURRENT_BINARY_DIR}" "" header "${header}")
      get_filename_component(headername ${header} NAME_WE)
      set(src ${CMAKE_CURRENT_BINARY_DIR}/testing/TestBuild_${name}_${headername}${suffix})
      configure_file(${Dax_SOURCE_DIR}/CMake/TestBuild.cxx.in ${src} @ONLY)
      list(APPEND cxxfiles ${src})
    endif()

  endforeach (header)

  #only attempt to add a test build executable if we have any headers to
  #test. this might not happen when everything depends on thrust.
  list(LENGTH cxxfiles cxxfiles_len)
  if (use_cuda AND ${cxxfiles_len} GREATER 0)
    cuda_add_library(TestBuild_${name} ${cxxfiles} ${hfiles})
  elseif (${cxxfiles_len} GREATER 0)
    add_library(TestBuild_${name} ${cxxfiles} ${hfiles})
    if(DAX_EXTRA_COMPILER_WARNINGS AND CMAKE_CXX_FLAGS_WARN_EXTRA)
      #only set the property if the user enabled warnings, and we have
      #a valid set of extra warnings to enable for the given compiler
      set_target_properties(TestBuild_${name}
        PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_WARN_EXTRA})
    endif(DAX_EXTRA_COMPILER_WARNINGS AND CMAKE_CXX_FLAGS_WARN_EXTRA)
  endif ()
  set_source_files_properties(${hfiles}
    PROPERTIES HEADER_FILE_ONLY TRUE
    )
endfunction(dax_add_header_build_test)

function(dax_install_headers dir_prefix)
  set(hfiles ${ARGN})
  install(FILES ${hfiles}
    DESTINATION ${Dax_INSTALL_INCLUDE_DIR}/${dir_prefix}
    )
endfunction(dax_install_headers)

# Declare a list of headers that require thrust to be enabled
# for them to header tested. In cases of thrust version 1.5 or less
# we have to make sure openMP is enabled, otherwise we are okay
function(dax_requires_thrust_to_test)
  #determine the state of thrust and testing
  set(cant_be_tested FALSE)
    if(NOT DAX_ENABLE_THRUST)
      #mark as not valid
      set(cant_be_tested TRUE)
    elseif(NOT DAX_ENABLE_OPENMP)
      #mark also as not valid
      set(cant_be_tested TRUE)
    endif()

  foreach(header ${ARGN})
    #set a property on the file that marks if we can header test it
    set_source_files_properties( ${header}
        PROPERTIES DAX_CANT_BE_HEADER_TESTED ${cant_be_tested} )

  endforeach(header)

endfunction(dax_requires_thrust_to_test)

# Declare a list of header files.  Will make sure the header files get
# compiled and show up in an IDE.
function(dax_declare_headers)
  set(options CUDA)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(DAX_DH "${options}"
    "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN}
    )
  set(hfiles ${DAX_DH_UNPARSED_ARGUMENTS})
  dax_get_kit_name(name dir_prefix)

  #only do header testing if enable testing is turned on
  if (DAX_ENABLE_TESTING)
    dax_add_header_build_test(
      "${name}" "${dir_prefix}" "${DAX_DH_CUDA}" ${hfiles})
  endif()
  #always install headers
  dax_install_headers("${dir_prefix}" ${hfiles})
endfunction(dax_declare_headers)

# Declare a list of worklet files.
function(dax_declare_worklets)
  # Currently worklets are just really header files.
  dax_declare_headers(${ARGN})
endfunction(dax_declare_worklets)

# Declare unit tests, which should be in the same directory as a kit
# (package, module, whatever you call it).  Usage:
#
# dax_unit_tests(
#   SOURCES <source_list>
#   LIBRARIES <dependent_library_list>
#   )
function(dax_unit_tests)
  set(options CUDA)
  set(oneValueArgs)
  set(multiValueArgs SOURCES LIBRARIES)
  cmake_parse_arguments(DAX_UT
    "${options}" "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN}
    )

  #set up what we possibly need to link too.
  list(APPEND DAX_UT_LIBRARIES ${TBB_LIBRARIES})
  #set up storage for the include dirs
  set(DAX_UT_INCLUDE_DIRS )

  if(DAX_ENABLE_OPENGL_INTEROP)
    list(APPEND DAX_UT_INCLUDE_DIRS ${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIR} )
    list(APPEND DAX_UT_LIBRARIES ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} )
  endif()

  if(DAX_ENABLE_OPENGL_TESTS)
    list(APPEND DAX_UT_INCLUDE_DIRS ${GLUT_INCLUDE_DIR} )
    list(APPEND DAX_UT_LIBRARIES ${GLUT_LIBRARIES}  )
  endif()

  if (DAX_ENABLE_TESTING)
    dax_get_kit_name(kit)
    #we use UnitTests_kit_ so that it is an unique key to exclude from coverage
    set(test_prog UnitTests_kit_${kit})
    create_test_sourcelist(TestSources ${test_prog}.cxx ${DAX_UT_SOURCES})
    if (DAX_UT_CUDA)
      cuda_add_executable(${test_prog} ${TestSources})
    else (DAX_UT_CUDA)
      add_executable(${test_prog} ${TestSources})
      if(DAX_EXTRA_COMPILER_WARNINGS AND CMAKE_CXX_FLAGS_WARN_EXTRA)
        set_target_properties(${test_prog}
          PROPERTIES COMPILE_FLAGS ${CMAKE_CXX_FLAGS_WARN_EXTRA})
      endif(DAX_EXTRA_COMPILER_WARNINGS AND CMAKE_CXX_FLAGS_WARN_EXTRA)
    endif (DAX_UT_CUDA)

    #do it as a property value so we don't pollute the include_directories
    #for any other targets
    set_property(TARGET ${test_prog} APPEND PROPERTY
        INCLUDE_DIRECTORIES ${DAX_UT_INCLUDE_DIRS} )

    target_link_libraries(${test_prog} ${DAX_UT_LIBRARIES})


    foreach (test ${DAX_UT_SOURCES})
      get_filename_component(tname ${test} NAME_WE)
      add_test(NAME ${tname}
        COMMAND ${test_prog} ${tname}
        )
    endforeach (test)
  endif (DAX_ENABLE_TESTING)
endfunction(dax_unit_tests)

# Save the worklets to test with each device adapter
# Usage:
#
# dax_save_worklet_unit_tests( sources )
#
# notes: will save the sources absolute path as the
# dax_source_worklet_unit_tests global property
function(dax_save_worklet_unit_tests )

  #create the test driver when we are called, since
  #the test driver expect the test files to be in the same
  #directory as the test driver
  create_test_sourcelist(test_sources WorkletTestDriver.cxx ${ARGN})

  #store the absolute path for the test drive and all the test
  #files
  set(driver ${CMAKE_CURRENT_BINARY_DIR}/WorkletTestDriver.cxx)
  set(cxx_sources)
  set(cu_sources)

  #we need to store the absolute source for the file so that
  #we can properly compile it into the test driver. At
  #the same time we want to configure each file into the build
  #directory as a .cu file so that we can compile it with cuda
  #if needed
  foreach(fname ${ARGN})
    set(absPath)

    get_filename_component(absPath ${fname} ABSOLUTE)
    get_filename_component(file_name_only ${fname} NAME_WE)

    set(cuda_file_name "${CMAKE_CURRENT_BINARY_DIR}/${file_name_only}.cu")
    configure_file("${absPath}"
                   "${cuda_file_name}"
                   COPYONLY)
    list(APPEND cxx_sources ${absPath})
    list(APPEND cu_sources ${cuda_file_name})
  endforeach()

  #we create a property that holds all the worklets to test,
  #but don't actually attempt to create a unit test with the yet.
  #That is done by each device adapter
  set_property( GLOBAL APPEND
                PROPERTY dax_worklet_unit_tests_sources ${cxx_sources})
  set_property( GLOBAL APPEND
                PROPERTY dax_worklet_unit_tests_cu_sources ${cu_sources})
  set_property( GLOBAL APPEND
                PROPERTY dax_worklet_unit_tests_drivers ${driver})

endfunction(dax_save_worklet_unit_tests)

# Call each worklet test for the given device adapter
# Usage:
#
# dax_worklet_unit_tests( device_adapter )
#
# notes: will look for the dax_source_worklet_unit_tests global
# property to find what are the worklet unit tests that need to be
# compiled for the give device adapter
function(dax_worklet_unit_tests device_adapter)

  set(unit_test_srcs)
  get_property(unit_test_srcs GLOBAL
               PROPERTY dax_worklet_unit_tests_sources )

  set(unit_test_drivers)
  get_property(unit_test_drivers GLOBAL
               PROPERTY dax_worklet_unit_tests_drivers )

  #detect if we are generating a .cu files
  set(is_cuda FALSE)
  set(old_nvcc_flags ${CUDA_NVCC_FLAGS})
  if("${device_adapter}" STREQUAL "DAX_DEVICE_ADAPTER_CUDA")
    set(is_cuda TRUE)
    #if we are generating cu files need to setup three things.
    #1. us the configured .cu files
    #2. Set BOOST_SP_DISABLE_THREADS to disable threading warnings
    #3. Disable unused function warnings
    #the FindCUDA module and helper methods don't read target level
    #properties so we have to modify CUDA_NVCC_FLAGS  instead of using
    # target and source level COMPILE_FLAGS and COMPILE_DEFINITIONS
    #
    get_property(unit_test_srcs GLOBAL PROPERTY dax_worklet_unit_tests_cu_sources )
    list(APPEND CUDA_NVCC_FLAGS -DBOOST_SP_DISABLE_THREADS)
    list(APPEND CUDA_NVCC_FLAGS "-w")
  endif()

  if(DAX_ENABLE_TESTING)
    dax_get_kit_name(kit)
    set(test_prog WorkletTests_${kit})

    if(is_cuda)
      cuda_add_executable(${test_prog} ${unit_test_drivers} ${unit_test_srcs})
    else()
      add_executable(${test_prog} ${unit_test_drivers} ${unit_test_srcs})
    endif()

    #add a test for each worklet test file. We will inject the device
    #adapter type into the test name so that it is easier to see what
    #exact device a test is failing on.

    string(REPLACE "DAX_DEVICE_ADAPTER_" "" device_type ${device_adapter})

    foreach (test ${unit_test_srcs})
      get_filename_component(tname ${test} NAME_WE)
      add_test(NAME "${tname}${device_type}"
        COMMAND ${test_prog} ${tname}
        )
    endforeach (test)

    #increase warning level if needed, we are going to skip cuda here
    #to remove all the false positive unused function warnings that cuda
    #generates
    if(DAX_EXTRA_COMPILER_WARNINGS AND CMAKE_CXX_FLAGS_WARN_EXTRA)
      set_property(TARGET ${test_prog}
            APPEND PROPERTY COMPILE_FLAGS ${CMAKE_CXX_FLAGS_WARN_EXTRA} )
    endif()

    #set the device adapter on the executable
    set_property(TARGET ${test_prog}
             APPEND
             PROPERTY COMPILE_DEFINITIONS DAX_DEVICE_ADAPTER=${device_adapter})
  endif()

  set(CUDA_NVCC_FLAGS ${old_nvcc_flags})
endfunction(dax_worklet_unit_tests)

# The Thrust project is not as careful as the Dax project in avoiding warnings
# on shadow variables and unused arguments.  With a real GCC compiler, you
# can disable these warnings inline, but with something like nvcc, those
# pragmas cause errors.  Thus, this macro will disable the compiler warnings.
macro(dax_disable_troublesome_thrust_warnings)
  dax_disable_troublesome_thrust_warnings_var(CMAKE_CXX_FLAGS_DEBUG)
  dax_disable_troublesome_thrust_warnings_var(CMAKE_CXX_FLAGS_MINSIZEREL)
  dax_disable_troublesome_thrust_warnings_var(CMAKE_CXX_FLAGS_RELEASE)
  dax_disable_troublesome_thrust_warnings_var(CMAKE_CXX_FLAGS_RELWITHDEBINFO)
endmacro(dax_disable_troublesome_thrust_warnings)

macro(dax_disable_troublesome_thrust_warnings_var flags_var)
  set(old_flags "${${flags_var}}")
  string(REPLACE "-Wshadow" "" new_flags "${old_flags}")
  string(REPLACE "-Wunused-parameter" "" new_flags "${new_flags}")
  string(REPLACE "-Wunused" "" new_flags "${new_flags}")
  string(REPLACE "-Wextra" "" new_flags "${new_flags}")
  string(REPLACE "-Wall" "" new_flags "${new_flags}")
  set(${flags_var} "${new_flags}")
endmacro(dax_disable_troublesome_thrust_warnings_var)

# Set up configuration for a given device.
macro(dax_configure_device device)
  string(TOUPPER "${device}" device_uppercase)
  set(Dax_ENABLE_${device_uppercase} ON)
  include("${Dax_SOURCE_DIR}/CMake/UseDax${device}.cmake")
  if(NOT Dax_${device}_FOUND)
    message(SEND_ERROR "Could not configure for using Dax with ${device}")
  endif(NOT Dax_${device}_FOUND)
endmacro(dax_configure_device)
