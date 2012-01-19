#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  CudaThreadIterator.h
  )

dax_declare_headers(CUDA ${headers})

set(unit_tests
  UnitTestCudaThreadIterator.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests})
