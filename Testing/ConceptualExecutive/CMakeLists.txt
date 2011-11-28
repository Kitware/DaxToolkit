#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})


#-----------------------------------------------------------------------------
set(sources
  Executive.cxx
  Executive.h
  )

cuda_add_library(daxExecutive SHARED ${sources})
target_link_libraries(daxExecutive
  ${BOOST_LIBRARIES}
  )


#-----------------------------------------------------------------------------
cuda_add_executable(FilterConcept
  main.cu
)

target_link_libraries(FilterConcept
  daxExecutive
  )


#-----------------------------------------------------------------------------
add_executable(FilterConcept2
  main2.cxx
)

target_link_libraries(FilterConcept2
  daxExecutive
  ${BOOST_LIBRARIES}
  )

