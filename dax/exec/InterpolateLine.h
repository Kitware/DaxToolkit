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

#ifndef __dax_exec_InterpolateLine_h
#define __dax_exec_InterpolateLine_h

#include <dax/exec/Cell.h>

#include <dax/exec/internal/InterpolationWeights.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------
template<typename ValueType,typename WeightType>
DAX_EXEC_EXPORT ValueType InterpolateLine(
    const ValueType &a,
    const ValueType &b,
    const WeightType &w)
{
  return a + w * (b-a);
}

}};

#endif //__dax_exec_InterpolateLine_h
