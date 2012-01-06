#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  ArrayContainerExecution.h
  CudaParameters.h
  DeviceArray.h
  WorkletFunctions.h
  )

dax_declare_headers(CUDA ${headers})

set(unit_tests
  UnitTestCudaArrayContainerExecution.cu
  UnitTestCudaDeviceArray.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests} LIBRARIES daxCont)
