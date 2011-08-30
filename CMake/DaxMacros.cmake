#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

function(dax_add_header_build_test name dir_prefix)
  set(HEADERS)
  set(hfiles)
  foreach (hdr ${ARGN})
    set(HEADERS "${HEADERS}#include <${dir_prefix}/${hdr}>\n")
    set(hfiles ${hfiles} ${Dax_SOURCE_DIR}/${dir_prefix}/${hdr})
  endforeach()
  
  configure_file(${Dax_SOURCE_DIR}/CMake/TestBuild.cxx.in
    ${CMAKE_CURRENT_BINARY_DIR}/TestBuild_${name}.cxx
    @ONLY)
  add_executable(TestBuild${name}
    ${CMAKE_CURRENT_BINARY_DIR}/TestBuild_${name}.cxx
    ${hfiles})
  set_source_files_properties(${hfiles}
    PROPERTIES HEADER_FILE_ONLY TRUE
    )
endfunction()

function(dax_declare_headers)
  set(hfiles ${ARGN})
  # Will this always work?  It should if ${CMAKE_CURRENT_SOURCE_DIR} is
  # built from ${Dax_SOURCE_DIR}.
  string(REPLACE "${Dax_SOURCE_DIR}/" "" dir_prefix ${CMAKE_CURRENT_SOURCE_DIR})
  string(REPLACE "/" "_" name "${dir_prefix}")
  dax_add_header_build_test("${name}" "${dir_prefix}" ${hfiles})
endfunction(dax_declare_headers)
