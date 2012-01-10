#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  )

#-----------------------------------------------------------------------------
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_subdirectory(internal)

#-----------------------------------------------------------------------------
#dax_declare_headers(CUDA ${headers})

#-----------------------------------------------------------------------------

add_subdirectory(worklet)
