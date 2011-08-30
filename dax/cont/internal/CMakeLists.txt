#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Macros.h
  SystemIncludes.h
  )

dax_declare_headers(${headers})

# These source files are actually compiled in the parent directory.
# They are in a seperate directory to highlight which objects are
# internal and which are part of the external interface.
add_custom_target(daxContPrivate ALL DEPENDS daxCont)
