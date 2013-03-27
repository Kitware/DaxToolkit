//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_internal_DeviceAdapterTag_h
#define __dax_cont_internal_DeviceAdapterTag_h

#include <dax/internal/Configure.h>
#include <dax/internal/ExportMacros.h>

#define DAX_DEVICE_ADAPTER_ERROR     -1
#define DAX_DEVICE_ADAPTER_UNDEFINED  0
#define DAX_DEVICE_ADAPTER_SERIAL     1
#define DAX_DEVICE_ADAPTER_CUDA       2
#define DAX_DEVICE_ADAPTER_OPENMP     3
#define DAX_DEVICE_ADAPTER_TBB        4

#ifndef DAX_DEVICE_ADAPTER
#ifdef DAX_CUDA
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_CUDA
#elif defined(DAX_OPENMP) // !DAX_CUDA
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_OPENMP
#elif defined(DAX_ENABLE_TBB) // !DAX_CUDA && !DAX_OPENMP
// Unfortunately, DAX_ENABLE_TBB does not guarantee that TBB is (or isn't)
// available, but there is no way to check for sure in a header library.
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_TBB
#else // !DAX_CUDA && !DAX_OPENMP && !DAX_ENABLE_TBB
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL
#endif // !DAX_CUDA && !DAX_OPENMP
#endif // DAX_DEVICE_ADAPTER

//-----------------------------------------------------------------------------
#if DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/internal/DeviceAdapterTagSerial.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cont::DeviceAdapterTagSerial

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_CUDA

#include <dax/cuda/cont/internal/DeviceAdapterTagCuda.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cuda::cont::DeviceAdapterTagCuda

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_OPENMP

#include <dax/openmp/cont/internal/DeviceAdapterTagOpenMP.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::openmp::cont::DeviceAdapterTagOpenMP

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_TBB

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::tbb::cont::DeviceAdapterTagTBB

#elif DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/internal/DeviceAdapterError.h>
#define DAX_DEFAULT_DEVICE_ADAPTER_TAG ::dax::cont::internal::DeviceAdapterTagError

#elif (DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_UNDEFINED) || !defined(DAX_DEVICE_ADAPTER)

#ifndef DAX_DEFAULT_DEVICE_ADAPTER_TAG
#warning If device adapter is undefined, DAX_DEFAULT_DEVICE_ADAPTER_TAG must be defined.
#endif

#else

#warning Unrecognized device adapter given.

#endif


#endif //__dax_cont_internal_DeviceAdapterTag_h
