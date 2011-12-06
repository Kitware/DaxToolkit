#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
include_directories(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
add_executable(PortsTest
  main.cxx
  FilterConnecters.h
  Filters.h
  Model.h
  Concept.h
  Worklets.h

)

target_link_libraries(PortsTest
  daxCont
  ${BOOST_LIBRARIES}
  )
