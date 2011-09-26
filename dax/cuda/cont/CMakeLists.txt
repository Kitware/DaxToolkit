#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(sources
  internal/DataArray.cu
  internal/DataBridge.cu
#  internal/KernelArgument.cu
  )

set(headers
  internal/DataArray.h
  internal/DataBridge.h
#  internal/KernelArgument.h
  )

#-----------------------------------------------------------------------------
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
cuda_add_library(daxCUDACont SHARED ${sources} ${headers})

set_source_files_properties(${headers}
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

target_link_libraries(daxCUDACont
  daxCont
  )

add_subdirectory(internal)
