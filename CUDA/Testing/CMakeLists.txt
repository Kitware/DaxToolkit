#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

#-----------------------------------------------------------------------------
CUDA_ADD_EXECUTABLE(DaxCUDATest
  DaxCUDATest.cu
  ArgumentsParser.cxx
  ArgumentsParser.h
)

target_link_libraries(DaxCUDATest
  daxCoreCont
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )

#-----------------------------------------------------------------------------
CUDA_ADD_EXECUTABLE(SimpleCUDAKernelTest
  SimpleCUDAKernelTest.cu
)

target_link_libraries(SimpleCUDAKernelTest
  daxCoreCont
  daxCont
  daxCUDACont 
  ${BOOST_LIBRARIES}
  )
