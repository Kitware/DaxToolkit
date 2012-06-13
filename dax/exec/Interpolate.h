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
template<class WorkType, class T, class ExecutionAdapter>
DAX_EXEC_EXPORT T cellInterpolate(
    const WorkType &work,
    const dax::exec::CellVoxel &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldPointIn<T, ExecutionAdapter> &point_field)
{
  const dax::Id numVerts = dax::exec::CellVoxel::NUM_POINTS;
  typedef dax::Tuple<T,numVerts> FieldTuple;

  dax::Scalar weights[numVerts];
  dax::exec::internal::interpolationWeightsVoxel(pcoords, weights);

  FieldTuple values = work.GetFieldValues(point_field);
  T result = 0;
  for (dax::Id vertexId = 0; vertexId < numVerts; vertexId++)
    {
    result += values[vertexId] * weights[vertexId];
    }

  return result;
}

}};

#endif //__dax_exec_Interpolate_h
