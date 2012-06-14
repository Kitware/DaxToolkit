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

#ifndef __dax_exec_Derivative_h
#define __dax_exec_Derivative_h

#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/math/LinearAlgebra.h>
#include <dax/exec/internal/DerivativeWeights.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellVoxel &cell,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &, //Not used for voxels
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &point_scalar)
{
  const dax::Id NUM_POINTS  = dax::exec::CellVoxel::NUM_POINTS;
  typedef dax::Tuple<dax::Vector3,NUM_POINTS> DerivWeights;

  DerivWeights derivativeWeights = dax::exec::internal::derivativeWeightsVoxel(
                                     pcoords);

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);
  dax::Tuple<dax::Scalar,NUM_POINTS> fieldValues =
      work.GetFieldValues(point_scalar);

  for (dax::Id vertexId = 0; vertexId < NUM_POINTS; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }

  return sum/cell.GetSpacing();
}

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellHexahedron &,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &fcoords,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &point_scalar)
{
  //for now we are considering that a cell hexahedron
  //is actually a voxel in an unstructured grid.
  //ToDo: use a proper derivative calculation.
  const dax::Id NUM_POINTS  = dax::exec::CellHexahedron::NUM_POINTS;
  typedef dax::Tuple<dax::Vector3,NUM_POINTS> DerivWeights;

  DerivWeights derivativeWeights = dax::exec::internal::derivativeWeightsVoxel(
                                     pcoords);

  dax::Vector3 spacing;
    {
    dax::Tuple<dax::Vector3,dax::exec::CellHexahedron::NUM_POINTS> allCoords;
    allCoords = work.GetFieldValues(fcoords);
    dax::Vector3 x0 = allCoords[0];
    dax::Vector3 x1 = allCoords[1];
    dax::Vector3 x2 = allCoords[2];
    dax::Vector3 x4 = allCoords[4];
    spacing = make_Vector3(x1[0] - x0[0],
                           x2[1] - x0[1],
                           x4[2] - x0[2]);
    }

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);
  dax::Tuple<dax::Scalar,NUM_POINTS> fieldValues =
      work.GetFieldValues(point_scalar);

  for (dax::Id vertexId = 0; vertexId < NUM_POINTS; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }


  return sum/spacing;
}

namespace internal {

//make this a seperate function so that these temporary values
//are scoped to be removed once we generate the matrix
DAX_EXEC_EXPORT dax::Tuple<dax::Vector2, 2> make_InvertedJacobian(
    dax::Vector3& len1,
    dax::Vector3& len2,
    const dax::Vector3& crossResult)
  {
  //ripped from VTK
  //dot product
  dax::Scalar lenX = dax::exec::math::Norm( len1 );
  dax::exec::math::Normalize(len1);
  dax::Vector2 dotResult(dax::dot(len2,len1),dax::dot(len2,crossResult));

  //invert the matrix drop b*c since b is zero
  dax::Scalar det = dax::Scalar(1.0)/(dotResult[1] * lenX);

  //compute the jacobian 2x2 matrix
  dax::Tuple<dax::Vector2, 2> invertedJacobain;
  invertedJacobain[0] = dax::make_Vector2(det * dotResult[1],
                                          det * (-dotResult[0]));
  invertedJacobain[1] = dax::make_Vector2(0,det * lenX);
  return invertedJacobain;
  }
}

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellTriangle &,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &fcoords,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &point_scalar)
{
  const dax::Id NUM_POINTS = dax::exec::CellTriangle::NUM_POINTS;
  typedef dax::Tuple<dax::Vector2,NUM_POINTS> DerivWeights;


  dax::Tuple<dax::Vector2, 2 > invertedJacobain;
  dax::Vector3 len1, len2;
  {
    const dax::Tuple<dax::Vector3,NUM_POINTS> x = work.GetFieldValues(fcoords);
    const dax::Vector3 crossResult = dax::normal(x[0],x[1],x[2]);
    len1 = x[1] - x[0];
    len2 = x[2] - x[0];
    invertedJacobain = dax::exec::internal::make_InvertedJacobian(len1,len2,
                                                                  crossResult);
  }

  DerivWeights derivativeWeights =
      dax::exec::internal::derivativeWeightsTriangle(pcoords);

  //compute sum
  dax::Vector2 sum = dax::make_Vector2(0.0, 0.0);
  dax::Tuple<dax::Scalar,NUM_POINTS> fieldValues =
      work.GetFieldValues(point_scalar);
  for (dax::Id vertexId = 0; vertexId < NUM_POINTS; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }

  //i think we can reorder these elements for better performance in the future
  dax::Vector2 dBy;
  dBy[0] = sum[0] * invertedJacobain[0][0] + sum[1] * invertedJacobain[0][1];
  dBy[1] = sum[0] * invertedJacobain[1][0] + sum[1] * invertedJacobain[1][1];

  len1 = len1 * dBy[0];
  len2 = len2 * dBy[1];
  return len1 + len2;
}


}};

#endif //__dax_exec_Derivative_h
