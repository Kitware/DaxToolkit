#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
project (DaxCommon)

include_directories(${Boost_INCLUDE_DIRS})

add_library(daxCommon SHARED
  daxObject.cxx
  daxObject.h
  daxMacros.h
  daxSystemIncludes.h)

target_link_libraries(daxCommon
  ${Boost_LIBRARIES}
  ${GOOGLE_CTEMPLATE_THREAD_LIB}
)

