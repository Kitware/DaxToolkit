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
//to define if Dax should be set to use 32bit data types. If you need to
//customize more of the dax type system, or what Device Adapters
//need to be included look at dax/internal/Configure.h for all defines that
//you can over-ride.

#ifndef __dax_internal_Configure32_h
#define __dax_internal_Configure32_h

#undef DAX_USE_DOUBLE_PRECISION
#undef DAX_USE_64BIT_IDS

#define DAX_SIZE_FLOAT 4
#define DAX_SIZE_INT 4

#include <dax/internal/Configure.h>

#endif