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

set(unit_tests
  UnitTestCudaArrayHandlePermutation.cu
  UnitTestCudaMath.cu
  UnitTestDeviceAdapterCuda.cu
  )
dax_unit_tests(CUDA SOURCES ${unit_tests})

#test all worklets with the CUDA device adapter
dax_worklet_unit_tests( DAX_DEVICE_ADAPTER_CUDA )