#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

#-----------------------------------------------------------------------------
CUDA_ADD_LIBRARY(daxCUDACont SHARED
  DataArray.cu
  DataArray.h
  DataBridge.cu
  DataBridge.h
  KernelArgument.cu
  KernelArgument.h
  )

target_link_libraries(daxCUDACont
  daxCont
  )
