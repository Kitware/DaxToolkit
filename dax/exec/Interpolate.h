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

#include <dax/CellTraits.h>
#include <dax/exec/CellField.h>
#include <dax/exec/internal/InterpolationWeights.h>
#include <dax/math/VectorAnalysis.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------
template<typename ValueType, class CellTag>
DAX_EXEC_EXPORT ValueType CellInterpolate(
    const dax::exec::CellField<ValueType,CellTag> &pointFieldValues,
    const dax::Vector3 &parametricCoords,
    CellTag)
{
  const dax::Id numVerts = dax::CellTraits<CellTag>::NUM_VERTICES;

  dax::Tuple<dax::Scalar, numVerts> weights =
      dax::exec::internal::InterpolationWeights(parametricCoords, CellTag());

  ValueType result = pointFieldValues[0] * weights[0];
  for (int vertexId = 1; vertexId < numVerts; vertexId++)
    {
    result = result + pointFieldValues[vertexId] * weights[vertexId];
    }

  return result;
}

//-----------------------------------------------------------------------------
template<typename ValueType,typename WeightType>
DAX_EXEC_EXPORT ValueType InterpolateLine(
    const ValueType &a,
    const ValueType &b,
    const WeightType &w)
{
  return dax::math::Lerp(a,b,w);
}
}};

#endif //__dax_exec_Interpolate_h
