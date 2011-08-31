#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-------------------------------------------------------------------------
# Find CUDA.
find_package(CUDA REQUIRED)
mark_as_advanced(CUDA_BUILD_CUBIN CUDA_BUILD_EMULATION CUDA_VERBOSE_BUILD)

add_subdirectory(internal)

add_subdirectory(exec)
add_subdirectory(cont)
