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

#ifndef __dax_exec_Interpolate_h
#define __dax_exec_Interpolate_h

#include <dax/exec/Cell.h>

#include <dax/exec/internal/InterpolationWeights.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------
template<typename ValueType, class CellType>
DAX_EXEC_EXPORT ValueType CellInterpolate(
    const CellType &daxNotUsed(cell),
    const dax::Tuple<ValueType,CellType::NUM_POINTS> &pointFieldValues,
    const dax::Vector3 &parametricCoords)
{
  const dax::Id numVerts = CellType::NUM_POINTS;

  dax::Tuple<dax::Scalar, numVerts> weights =
      dax::exec::internal::InterpolationWeights<CellType>(parametricCoords);

  ValueType result = pointFieldValues[0] * weights[0];
  for (dax::Id vertexId = 1; vertexId < numVerts; vertexId++)
    {
    result = result + pointFieldValues[vertexId] * weights[vertexId];
    }

  return result;
}

}};

#endif //__dax_exec_Interpolate_h
