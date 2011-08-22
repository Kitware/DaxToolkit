#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
include_directories(${Boost_INCLUDE_DIRS})

add_library(daxCoreCont SHARED
  Object.cxx
  Object.h
  SystemIncludes.h
  Macros.h
)

target_link_libraries(daxCoreCont
  ${Boost_LIBRARIES})
