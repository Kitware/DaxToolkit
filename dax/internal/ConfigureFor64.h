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
//This header can be used by external application that are consuming Dax
//to define if Dax should be set to use 64bit data types. If you need to
//customize more of the dax type system, or what Device Adapters
//need to be included look at dax/internal/Configure.h for all defines that
//you can over-ride.
#ifdef __dax_internal_Configure_h
# error Incorrect header include order. Include this header first
#endif

#ifndef __dax_internal_Configure32_h
#define __dax_internal_Configure32_h

#define DAX_USE_DOUBLE_PRECISION
#define DAX_USE_64BIT_IDS

#include <dax/internal/Configure.h>

#endif