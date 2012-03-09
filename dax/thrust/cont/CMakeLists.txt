#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  LowerBounds.h
  ScheduleThrust.h
  Sort.h
  StreamCompact.h
  Unique.h
  )

#-----------------------------------------------------------------------------
add_subdirectory(internal)

#-----------------------------------------------------------------------------
dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  )
#dax_unit_tests(SOURCES ${unit_tests})
