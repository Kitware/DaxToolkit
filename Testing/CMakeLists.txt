#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

if (DAX_ENABLE_CUDA)
  add_subdirectory(cuda)
endif (DAX_ENABLE_CUDA)

add_subdirectory(ConceptualExecutive)
