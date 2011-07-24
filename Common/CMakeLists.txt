#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================
project (DaxCommon)

include_directories(${Boost_INCLUDE_DIRS})

CUDA_ADD_LIBRARY(daxCommon SHARED
  DaxDataArray.cu
  DaxDataArray.h
  DaxDataSet.h
  DaxKernelArgument.h
  daxMacros.h
  daxObject.cxx
  daxObject.h
  daxSystemIncludes.h
  )


target_link_libraries(daxCommon
  ${Boost_LIBRARIES}
  ${GOOGLE_CTEMPLATE_THREAD_LIB}
)

