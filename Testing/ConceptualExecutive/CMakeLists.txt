#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
#CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
add_executable(FilterConcept
  main.cxx
)

target_link_libraries(FilterConcept
  ${BOOST_LIBRARIES}
  )
