#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-------------------------------------------------------------------------
# Thrust comes with the latest versions of CUDA.  Find CUDA first if it
# is enabled, which will simplify finding thrust.
if (DAX_ENABLE_CUDA)
  find_package(CUDA)
  mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)
endif (DAX_ENABLE_CUDA)

find_package(Thrust REQUIRED)

add_subdirectory(cont)
