#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

function (add_header_build_test name dir_prefix)
  set (HEADERS)
  foreach (hdr ${ARGN})
    set (HEADERS "${HEADERS}#include \"${dir_prefix}/${hdr}\"\n")
  endforeach()
  
  configure_file(${Dax_SOURCE_DIR}/CMake/TestBuild.cxx.in
    ${CMAKE_CURRENT_BINARY_DIR}/TestBuild${name}.cxx
    @ONLY)
  add_executable(TestBuild${name}
    ${CMAKE_CURRENT_BINARY_DIR}/TestBuild${name}.cxx)
endfunction()

