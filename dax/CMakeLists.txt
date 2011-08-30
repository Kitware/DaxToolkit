#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Types.h
  )

dax_declare_headers(${headers})

add_subdirectory(internal)

add_subdirectory(cont)
add_subdirectory(exec)
