#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
find_package(Doxygen REQUIRED)

#-----------------------------------------------------------------------------
# Configure the options file.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.config.in
  ${CMAKE_CURRENT_BINARY_DIR}/doxygen.config @ONLY)

add_custom_target(documentation ALL
  COMMAND ${DOXYGEN_EXECUTABLE}
          ${CMAKE_CURRENT_BINARY_DIR}/doxygen.config
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

