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
cuda_add_executable(FilterConcept
  main.cu
)

target_link_libraries(FilterConcept
  ${BOOST_LIBRARIES}
  )
