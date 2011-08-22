#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
add_subdirectory(Common)
add_subdirectory(Control)
add_subdirectory(Execution)

if (DAX_ENABLE_TESTING)
  add_subdirectory(Testing)
endif()
