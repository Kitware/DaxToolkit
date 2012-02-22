#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  DeviceAdapterOpenMP.h
  ReIndexThrust.h
  ScheduleThrust.h
  )

#-----------------------------------------------------------------------------
include_directories(${THRUST_INCLUDE_DIR})

add_subdirectory(internal)

#-----------------------------------------------------------------------------
dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestDeviceAdapterOpenMP.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
