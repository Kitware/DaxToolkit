##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

set(headers
  Copy.h
  DeviceAdapterCuda.h
  InclusiveScan.h
  LowerBounds.h
  ScheduleCuda.h
  ScheduleThrust.h
  StreamCompact.h
  Sort.h
  Unique.h
  )

#-----------------------------------------------------------------------------
CUDA_INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_subdirectory(internal)

#-----------------------------------------------------------------------------
dax_declare_headers(CUDA ${headers})

#-----------------------------------------------------------------------------
set(unit_tests
  UnitTestCudaMath.cu
  UnitTestCudaScheduleNative.cu
  UnitTestCudaScheduleThrust.cu
  UnitTestDeviceAdapterCuda.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests})
