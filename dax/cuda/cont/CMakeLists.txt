#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  DeviceAdapterCuda.h
  ScheduleCuda.h
  ScheduleThrust.h
  )

#-----------------------------------------------------------------------------
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_subdirectory(internal)

#-----------------------------------------------------------------------------
dax_declare_headers(CUDA ${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestCudaScheduleNative.cu
  UnitTestCudaScheduleThrust.cu
  UnitTestCudaWorkletMapCell.cu
  UnitTestCudaWorkletMapField.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests})
