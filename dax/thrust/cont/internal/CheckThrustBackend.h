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
#ifndef __dax_thrust_cont_internal_CheckThrustBackend_h
#define __dax_thrust_cont_internal_CheckThrustBackend_h

// Dax thrust headers should only be included by other header files using thrust
// with a specific device, which should always set the thrust backend first.
// The only exception is the header test compiles.  Issue a warning if someone
// inappropriately includes this file.

#include <dax/internal/Configure.h>


#ifdef DAX_ENABLE_THRUST

#if DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION >= 6

// Set the backend to be serial so it will work everywhere,
// this was only added in thrust 1.7 and later
#ifndef THRUST_DEVICE_SYSTEM
# ifdef DAX_TEST_HEADER_BUILD
#   define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
# else // DAX_TEST_HEADER_BUILD
#   error Inappropriate use of thrust headers.
# endif // DAX_TEST_HEADER_BUILD
#endif //THRUST_DEVICE_SYSTEM

#elif DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION <= 5

//we are going to try and pick open mp if it is enabled
#ifndef THRUST_DEVICE_BACKEND
# ifdef DAX_TEST_HEADER_BUILD
#   define THRUST_DEVICE_BACKEND THRUST_DEVICE_BACKEND_OMP
# else // DAX_TEST_HEADER_BUILD
#   error Inappropriate use of thrust headers.
# endif // DAX_TEST_HEADER_BUILD
#endif //THRUST_DEVICE_BACKEND

#endif //DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION <= 5


#endif //DAX_ENABLE_THRUST


#endif // __dax_thrust_cont_internal_CheckThrustBackend_h
