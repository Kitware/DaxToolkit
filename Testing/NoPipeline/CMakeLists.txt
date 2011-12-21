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
cuda_add_executable(NoPipelineTest
  main.cu
)

target_link_libraries(NoPipelineTest
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )
