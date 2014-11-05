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
#ifndef __dax_cuda_cont_internal_SetThrustForCuda_h
#define __dax_cuda_cont_internal_SetThrustForCuda_h

#include <dax/internal/Configure.h>

#ifdef DAX_ENABLE_THRUST

#if DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION >= 6

#ifndef THRUST_DEVICE_SYSTEM
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA
#else // defined THRUST_DEVICE_BACKEND
#if THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#error Thrust device backend set incorrectly.
#endif // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#endif // defined(THRUST_DEVICE_SYSTEM)


#else //DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION >= 6

#ifndef THRUST_DEVICE_BACKEND
#define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_CUDA
#else // defined THRUST_DEVICE_BACKEND
#if THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
#error Thrust device backend set incorrectly.
#endif // THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
#endif // defined THRUST_DEVICE_BACKEND


#endif //DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION >= 6



#endif //DAX_ENABLE_THRUST


#endif //__dax_cuda_cont_internal_SetThrustForCuda_h
