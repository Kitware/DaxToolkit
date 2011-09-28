#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(unit_tests
  UnitTestCudaManagedDeviceDataArray.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests} LIBRARIES daxCont)
