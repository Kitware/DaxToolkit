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
template<class WorkType, class FieldType>
DAX_EXEC_EXPORT typename FieldType::ValueType CellInterpolate(
    const WorkType &work,
    const typename WorkType::CellType &, // Should we get rid of the parameter?
    const dax::Vector3 &pcoords,
    const FieldType &point_field)
{
  typedef typename WorkType::CellType CellType;
  typedef typename FieldType::ValueType ValueType;
  const dax::Id numVerts = CellType::NUM_POINTS;
  typedef dax::Tuple<ValueType,numVerts> FieldTuple;

  dax::Tuple<dax::Scalar, numVerts> weights =
      dax::exec::internal::InterpolationWeights<CellType>(pcoords);

  FieldTuple values = work.GetFieldValues(point_field);
  ValueType result = values[0] * weights[0];
  for (dax::Id vertexId = 1; vertexId < numVerts; vertexId++)
    {
    result = result + values[vertexId] * weights[vertexId];
    }

  return result;
}

}};

#endif //__dax_exec_Interpolate_h
