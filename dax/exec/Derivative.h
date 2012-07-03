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
#include <dax/exec/math/Matrix.h>
#include <dax/exec/math/VectorAnalysis.h>
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
namespace detail {
// This returns the Jacobian of a hexahedron's coordinates with respect to
// parametric coordinates.  Explicitly, this is (d is partial derivative):
//
//   |                     |
//   | dx/du  dx/dv  dx/dw |
//   |                     |
//   | dy/du  dy/dv  dy/dw |
//   |                     |
//   | dz/du  dz/dv  dz/dw |
//   |                     |
//
DAX_EXEC_EXPORT
dax::exec::math::Matrix3x3 make_JacobianForHexahedron(
    const dax::Tuple<dax::Vector3,dax::exec::CellHexahedron::NUM_POINTS>
    &derivativeWeights,
    const dax::Tuple<dax::Vector3,dax::exec::CellHexahedron::NUM_POINTS>
    &pointCoordinates)
{
  const int NUM_POINTS = dax::exec::CellHexahedron::NUM_POINTS;
  dax::exec::math::Matrix3x3 jacobian(0);
  for (int pointIndex = 0; pointIndex < NUM_POINTS; pointIndex++)
    {
    const dax::Vector3 &dweight = derivativeWeights[pointIndex];
    const dax::Vector3 &pcoord = pointCoordinates[pointIndex];

    jacobian(0,0) += pcoord[0] * dweight[0];
    jacobian(0,1) += pcoord[0] * dweight[1];
    jacobian(0,2) += pcoord[0] * dweight[2];

    jacobian(1,0) += pcoord[1] * dweight[0];
    jacobian(1,1) += pcoord[1] * dweight[1];
    jacobian(1,2) += pcoord[1] * dweight[2];

    jacobian(2,0) += pcoord[2] * dweight[0];
    jacobian(2,1) += pcoord[2] * dweight[1];
    jacobian(2,2) += pcoord[2] * dweight[2];
    }

  return jacobian;
}
}

template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellHexahedron &,
    const dax::Vector3 &pcoords,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &fcoords,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &point_scalar)
{
  const dax::Id NUM_POINTS  = dax::exec::CellHexahedron::NUM_POINTS;
  typedef dax::Tuple<dax::Vector3,NUM_POINTS> DerivWeights;

  DerivWeights derivativeWeights =
      dax::exec::internal::derivativeWeightsHexahedron(pcoords);
  dax::Tuple<dax::Vector3,dax::exec::CellHexahedron::NUM_POINTS> allCoords =
      work.GetFieldValues(fcoords);

  // For reasons that should become apparent in a moment, we actually want
  // the transpose of the Jacobian.
  dax::exec::math::Matrix3x3 jacobianTranspose =
      dax::exec::math::MatrixTranspose(
        detail::make_JacobianForHexahedron(derivativeWeights, allCoords));

  // Find the derivative of the field in parametric coordinate space. That is,
  // find the vector [ds/du, ds/dv, ds/dw].
  dax::Tuple<dax::Scalar,NUM_POINTS> fieldValues =
      work.GetFieldValues(point_scalar);
  dax::Vector3 parametricDerivative(dax::Scalar(0));
  for (int pointIndex = 0; pointIndex < NUM_POINTS; pointIndex++)
    {
    parametricDerivative =
        parametricDerivative
        + derivativeWeights[pointIndex] * fieldValues[pointIndex];
    }

  // If we write out the matrices below, it should become clear that the
  // Jacobian transpose times the field derivative in world space equals
  // the field derivative in parametric space.
  //
  //   |                     |  |       |     |       |
  //   | dx/du  dy/du  dz/du |  | ds/dx |     | ds/du |
  //   |                     |  |       |     |       |
  //   | dx/dv  dy/dv  dz/dv |  | ds/dy |  =  | ds/dv |
  //   |                     |  |       |     |       |
  //   | dx/dw  dy/dw  dz/dw |  | ds/dz |     | ds/dw |
  //   |                     |  |       |     |       |
  //
  // Now we just need to solve this linear system to find the derivative in
  // world space.

  bool valid;  // Ignored.
  return dax::exec::math::SolveLinearSystem(jacobianTranspose,
                                            parametricDerivative,
                                            valid);
}


//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellTetrahedron&/*cell*/,
    const dax::Vector3 &/*pcoords*/,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &scalarField)
{
  const dax::Id NUM_POINTS = dax::exec::CellTetrahedron::NUM_POINTS;

  // The scalar values of the four points in a tetrahedron completely specify a
  // linear field (with constant gradient). The field, defined by the 3-vector
  // gradient g and scalar value s_origin, can be found with this set of 4
  // equations and 4 unknowns.
  //
  // dot(p0, g) + s_origin = s0
  // dot(p1, g) + s_origin = s1
  // dot(p2, g) + s_origin = s2
  // dot(p3, g) + s_origin = s3
  //
  // Where the p's are point coordinates. But we don't really care about
  // s_origin. We just want to find the gradient g. With some simple
  // elimination we, we can get rid of s_origin and be left with 3 equations
  // and 3 unknowns.
  //
  // dot(p1-p0, g) = s1 - s0
  // dot(p2-p0, g) = s2 - s0
  // dot(p3-p0, g) = s3 - s0
  //
  // We'll solve this by putting this in matrix form Ax = b where the rows of A
  // are the differences in points and normal, b has the scalar differences,
  // and x is really the gradient g.
  //
  dax::Tuple<dax::Vector3, NUM_POINTS> p = work.GetFieldValues(coordField);
  dax::Vector3 v0 = p[1] - p[0];
  dax::Vector3 v1 = p[2] - p[0];
  dax::Vector3 v2 = p[3] - p[0];

  dax::exec::math::Matrix3x3 A;
  dax::exec::math::MatrixSetRow(A, 0, v0);
  dax::exec::math::MatrixSetRow(A, 1, v1);
  dax::exec::math::MatrixSetRow(A, 2, v2);

  dax::Tuple<dax::Scalar, NUM_POINTS> s = work.GetFieldValues(scalarField);
  dax::Vector3 b(s[1]-s[0], s[2]-s[0], s[3]-s[0]);

  // If we want to later change this method to take the gradient of multiple
  // values (for example, to find the Jacobian of a vector field), then there
  // are more efficient ways solve them all than independently solving this
  // equation for each component of the field.  You could find the inverse of
  // matrix A.  Or you could alter the functions in dax::exec::math to
  // simultaneously solve multiple equations.

  // If the tetrahedron is degenerate, then valid will be false. For now we are
  // ignoring it. We could detect it if we determine we need to although I have
  // seen singular matrices missed due to floating point error.
  //
  bool valid;

  return dax::exec::math::SolveLinearSystem(A, b, valid);
}


//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 cellDerivative(
    const WorkType &work,
    const dax::exec::CellTriangle &/*cell*/,
    const dax::Vector3 &/*pcoords*/,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &scalarField)
{
  const dax::Id NUM_POINTS = dax::exec::CellTriangle::NUM_POINTS;

  // The scalar values of the three points in a triangle completely specify a
  // linear field (with constant gradient) assuming the field is constant in
  // the normal direction to the triangle. The field, defined by the 3-vector
  // gradient g and scalar value s_origin, can be found with this set of 4
  // equations and 4 unknowns.
  //
  // dot(p0, g) + s_origin = s0
  // dot(p1, g) + s_origin = s1
  // dot(p2, g) + s_origin = s2
  // dot(n, g)             = 0
  //
  // Where the p's are point coordinates and n is the normal vector. But we
  // don't really care about s_origin. We just want to find the gradient g.
  // With some simple elimination we, we can get rid of s_origin and be left
  // with 3 equations and 3 unknowns.
  //
  // dot(p1-p0, g) = s1 - s0
  // dot(p2-p0, g) = s2 - s0
  // dot(n, g)     = 0
  //
  // We'll solve this by putting this in matrix form Ax = b where the rows of A
  // are the differences in points and normal, b has the scalar differences,
  // and x is really the gradient g.
  //
  dax::Tuple<dax::Vector3, NUM_POINTS> p = work.GetFieldValues(coordField);
  dax::Vector3 v0 = p[1] - p[0];
  dax::Vector3 v1 = p[2] - p[0];
  dax::Vector3 n = dax::exec::math::Cross(v0, v1);

  dax::exec::math::Matrix3x3 A;
  dax::exec::math::MatrixSetRow(A, 0, v0);
  dax::exec::math::MatrixSetRow(A, 1, v1);
  dax::exec::math::MatrixSetRow(A, 2, n);

  dax::Tuple<dax::Scalar, NUM_POINTS> s = work.GetFieldValues(scalarField);
  dax::Vector3 b(s[1]-s[0], s[2]-s[0], 0);

  // If we want to later change this method to take the gradient of multiple
  // values (for example, to find the Jacobian of a vector field), then there
  // are more efficient ways solve them all than independently solving this
  // equation for each component of the field.  You could find the inverse of
  // matrix A.  Or you could alter the functions in dax::exec::math to
  // simultaneously solve multiple equations.

  // If the triangle is degenerate, then valid will be false.  For now we are
  // ignoring it.  We could detect it if we determine we need to although I
  // have seen singular matrices missed due to floating point error.
  //
  bool valid;

  return dax::exec::math::SolveLinearSystem(A, b, valid);
}

}};

#endif //__dax_exec_Derivative_h
