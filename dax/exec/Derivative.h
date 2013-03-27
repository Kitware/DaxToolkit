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

#include <dax/CellTag.h>
#include <dax/CellTraits.h>

#include <dax/exec/CellField.h>

#include <dax/exec/internal/DerivativeWeights.h>

#include <dax/math/Matrix.h>
#include <dax/math/VectorAnalysis.h>

namespace dax { namespace exec {

//-----------------------------------------------------------------------------

/// Special version of CellDerivative for Voxels or other axis aligned cells
/// that do not require vertex coordinates.
///
template<class CellTag>
DAX_EXEC_EXPORT dax::Vector3 CellDerivativeAxisAligned(
    const dax::Vector3 &parametricCoords,
    const dax::Vector3 &axisAlignedWidths,
    const dax::exec::CellField<dax::Scalar, CellTag> &fieldValues,
    CellTag)
{
  const dax::Id NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  typedef dax::Tuple<dax::Vector3,NUM_VERTICES> DerivWeights;

  DerivWeights derivativeWeights =
      dax::exec::internal::DerivativeWeights(parametricCoords, CellTag());

  dax::Vector3 sum = dax::make_Vector3(0.0, 0.0, 0.0);

  for (int vertexId = 0; vertexId < NUM_VERTICES; vertexId++)
    {
    sum = sum + fieldValues[vertexId] * derivativeWeights[vertexId];
    }

  return sum/axisAlignedWidths;
}

/// If you can get the axis aligned widths (i.e. spacing) of the voxel rather
/// than the vertex coordinates, than using the CellDerivativeAxisAligned
/// version would be more efficient.
///
DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &parametricCoords,
    const dax::exec::CellField<dax::Vector3, dax::CellTagVoxel> &vertCoords,
    const dax::exec::CellField<dax::Scalar, dax::CellTagVoxel> &fieldValues,
    dax::CellTagVoxel)
{
  dax::Vector3 axisAlignedWidths = vertCoords[6] - vertCoords[0];
  return CellDerivativeAxisAligned(parametricCoords,
                                   axisAlignedWidths,
                                   fieldValues,
                                   dax::CellTagVoxel());
}

//-----------------------------------------------------------------------------
namespace detail {

// This returns the Jacobian of a hexahedron's (or other 3D cell's coordinates
// with respect to parametric coordinates. Explicitly, this is (d is partial
// derivative):
//
//   |                     |
//   | dx/du  dx/dv  dx/dw |
//   |                     |
//   | dy/du  dy/dv  dy/dw |
//   |                     |
//   | dz/du  dz/dv  dz/dw |
//   |                     |
//
template<int NUM_VERTICES>
DAX_EXEC_EXPORT
dax::math::Matrix3x3 make_JacobianFor3DCell(
    const dax::Tuple<dax::Vector3,NUM_VERTICES> &derivativeWeights,
    const dax::Tuple<dax::Vector3,NUM_VERTICES> &vertCoords)
{
  dax::math::Matrix3x3 jacobian(0);
  for (int pointIndex = 0; pointIndex < NUM_VERTICES; pointIndex++)
    {
    const dax::Vector3 &dweight = derivativeWeights[pointIndex];
    const dax::Vector3 &pcoord = vertCoords[pointIndex];

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

template<class CellTag>
DAX_EXEC_EXPORT dax::Vector3 CellDerivativeFor3DCell(
    const dax::Vector3 &parametricCoords,
    const dax::exec::CellField<dax::Vector3, CellTag> &vertCoords,
    const dax::exec::CellField<dax::Scalar, CellTag> &fieldValues,
    CellTag)
{
  const dax::Id NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  typedef dax::Tuple<dax::Vector3,NUM_VERTICES> DerivWeights;

  DerivWeights derivativeWeights =
      dax::exec::internal::DerivativeWeights(parametricCoords, CellTag());

  // For reasons that should become apparent in a moment, we actually want
  // the transpose of the Jacobian.
  dax::math::Matrix3x3 jacobianTranspose =
      dax::math::MatrixTranspose(
        detail::make_JacobianFor3DCell(derivativeWeights,
                                       vertCoords.GetAsTuple()));

  // Find the derivative of the field in parametric coordinate space. That is,
  // find the vector [ds/du, ds/dv, ds/dw].
  dax::Vector3 parametricDerivative(dax::Scalar(0));
  for (int pointIndex = 0; pointIndex < NUM_VERTICES; pointIndex++)
    {
    parametricDerivative =
        parametricDerivative
        + (fieldValues[pointIndex] * derivativeWeights[pointIndex]);
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
  return dax::math::SolveLinearSystem(jacobianTranspose,
                                      parametricDerivative,
                                      valid);
}

}

DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &parametricCoords,
    const dax::exec::CellField<dax::Vector3,dax::CellTagHexahedron> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagHexahedron> &fieldValues,
    dax::CellTagHexahedron)
{
  return detail::CellDerivativeFor3DCell(
        parametricCoords, vertCoords, fieldValues, dax::CellTagHexahedron());
}

DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &parametricCoords,
    const dax::exec::CellField<dax::Vector3,dax::CellTagWedge> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagWedge> &fieldValues,
    dax::CellTagWedge)
{
  return detail::CellDerivativeFor3DCell(
        parametricCoords, vertCoords, fieldValues, dax::CellTagWedge());
}


//-----------------------------------------------------------------------------
DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &daxNotUsed(parametricCoords),
    const dax::exec::CellField<dax::Vector3,dax::CellTagTetrahedron> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagTetrahedron> &fieldValues,
    dax::CellTagTetrahedron)
{
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
  dax::Vector3 v0 = vertCoords[1] - vertCoords[0];
  dax::Vector3 v1 = vertCoords[2] - vertCoords[0];
  dax::Vector3 v2 = vertCoords[3] - vertCoords[0];

  dax::math::Matrix3x3 A;
  dax::math::MatrixSetRow(A, 0, v0);
  dax::math::MatrixSetRow(A, 1, v1);
  dax::math::MatrixSetRow(A, 2, v2);

  dax::Vector3 b(fieldValues[1]-fieldValues[0],
                 fieldValues[2]-fieldValues[0],
                 fieldValues[3]-fieldValues[0]);

  // If we want to later change this method to take the gradient of multiple
  // values (for example, to find the Jacobian of a vector field), then there
  // are more efficient ways solve them all than independently solving this
  // equation for each component of the field.  You could find the inverse of
  // matrix A.  Or you could alter the functions in dax::math to
  // simultaneously solve multiple equations.

  // If the tetrahedron is degenerate, then valid will be false. For now we are
  // ignoring it. We could detect it if we determine we need to although I have
  // seen singular matrices missed due to floating point error.
  //
  bool valid;

  return dax::math::SolveLinearSystem(A, b, valid);
}


//-----------------------------------------------------------------------------
DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &daxNotUsed(parametricCoords),
    const dax::exec::CellField<dax::Vector3,dax::CellTagTriangle> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagTriangle> &fieldValues,
    dax::CellTagTriangle)
{
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
  dax::Vector3 v0 = vertCoords[1] - vertCoords[0];
  dax::Vector3 v1 = vertCoords[2] - vertCoords[0];
  dax::Vector3 n = dax::math::Cross(v0, v1);

  dax::math::Matrix3x3 A;
  dax::math::MatrixSetRow(A, 0, v0);
  dax::math::MatrixSetRow(A, 1, v1);
  dax::math::MatrixSetRow(A, 2, n);

  dax::Vector3 b(fieldValues[1]-fieldValues[0],
                 fieldValues[2]-fieldValues[0],
                 0);

  // If we want to later change this method to take the gradient of multiple
  // values (for example, to find the Jacobian of a vector field), then there
  // are more efficient ways solve them all than independently solving this
  // equation for each component of the field.  You could find the inverse of
  // matrix A.  Or you could alter the functions in dax::math to
  // simultaneously solve multiple equations.

  // If the triangle is degenerate, then valid will be false.  For now we are
  // ignoring it.  We could detect it if we determine we need to although I
  // have seen singular matrices missed due to floating point error.
  //
  bool valid;

  return dax::math::SolveLinearSystem(A, b, valid);
}

//-----------------------------------------------------------------------------
namespace detail {

struct QuadrilateralSpace
{
  dax::Vector3 Origin;
  dax::Vector3 Basis0;
  dax::Vector3 Basis1;
};

// The gradient is overconstrained for a quadrilateral. To get around the
// problem, we build a 2D space (different from parametric space) in the same
// plane as the quadrilateral and is linearlly consistant so that we can
// validly compute the gradient in this space and convert back to real space.
// For convienience, we also put the origin at the first point of the
// quadrilateral and place the "x" axis on the first edge of the quadrilateral.
//
DAX_EXEC_EXPORT
void make_SpaceForQuadrilateral(
    const dax::Tuple<dax::Vector3, dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES> &vertCoords,
    QuadrilateralSpace &space,
    dax::Tuple<dax::Vector2, dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES> &vertCoords2D)
{
  space.Origin = vertCoords[0];

  dax::Vector3 vec1 = vertCoords[1] - space.Origin;
  dax::Scalar lengthX = dax::math::Magnitude(vec1);
  space.Basis0 = vec1 * (1/lengthX);

  dax::Vector3 vec2 = vertCoords[2] - space.Origin;
  dax::Vector3 normal = dax::math::Cross(vec1, vec2);
  space.Basis1 =
      dax::math::Normal(dax::math::Cross(space.Basis0, normal));

  vertCoords2D[0] = dax::make_Vector2(0, 0);
  vertCoords2D[1] = dax::make_Vector2(lengthX, 0);
  vertCoords2D[2] = dax::make_Vector2(dax::dot(space.Basis0, vec2),
                                            dax::dot(space.Basis1, vec2));
  dax::Vector3 vec3 = vertCoords[3] - space.Origin;
  vertCoords2D[3] = dax::make_Vector2(dax::dot(space.Basis0, vec3),
                                            dax::dot(space.Basis1, vec3));
}

// Derivatives in quadrilaterals are computed in much the same way as
// hexahedra.  Review the documentation for hexahedra derivatives for details
// on the math.  The major difference is that the equations are performed in
// a 2D space built with make_SpaceForQuadrilateral.
//
DAX_EXEC_EXPORT
dax::math::Matrix2x2 make_JacobianForQuadrilateral(
    const dax::Tuple<dax::Vector3, dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES> &derivativeWeights,
    const dax::Tuple<dax::Vector2, dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES> &vertCoords2D)
{
  const int NUM_VERTICES =
      dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES;

  dax::math::Matrix2x2 jacobian(0);

  for (int pointIndex = 0; pointIndex < NUM_VERTICES; pointIndex++)
    {
    const dax::Vector3 &dweight = derivativeWeights[pointIndex];
    const dax::Vector2 coord = vertCoords2D[pointIndex];

    jacobian(0,0) += coord[0] * dweight[0];
    jacobian(0,1) += coord[0] * dweight[1];

    jacobian(1,0) += coord[1] * dweight[0];
    jacobian(1,1) += coord[1] * dweight[1];
    }

  return jacobian;
}

} // namespace detail

DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &parametricCoords,
    const dax::exec::CellField<dax::Vector3,dax::CellTagQuadrilateral> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagQuadrilateral> &fieldValues,
    dax::CellTagQuadrilateral)
{
  const dax::Id NUM_VERTICES =
      dax::CellTraits<dax::CellTagQuadrilateral>::NUM_VERTICES;
  typedef dax::Tuple<dax::Vector3,NUM_VERTICES> DerivWeights;

  DerivWeights derivativeWeights =
      dax::exec::internal::DerivativeWeights(
        parametricCoords, dax::CellTagQuadrilateral());

  // We have an overconstrained system, so create a 2D space in the plane
  // that the quadrilateral sits.

  detail::QuadrilateralSpace space;
  dax::Tuple<dax::Vector2,NUM_VERTICES> vertCoords2D;
  detail::make_SpaceForQuadrilateral(
        vertCoords.GetAsTuple(), space, vertCoords2D);

  // For reasons that should become apparent in a moment, we actually want
  // the transpose of the Jacobian.
  dax::math::Matrix2x2 jacobianTranspose =
      dax::math::MatrixTranspose(
        detail::make_JacobianForQuadrilateral(derivativeWeights,vertCoords2D));

  // Find the derivative of the field in parametric coordinate space. That is,
  // find the vector [ds/du, ds/dv].
  dax::Vector2 parametricDerivative(dax::Scalar(0));
  for (int pointIndex = 0; pointIndex < NUM_VERTICES; pointIndex++)
    {
    parametricDerivative[0] +=
        derivativeWeights[pointIndex][0] * fieldValues[pointIndex];
    parametricDerivative[1] +=
        derivativeWeights[pointIndex][1] * fieldValues[pointIndex];
    }

  // If we write out the matrices below, it should become clear that the
  // Jacobian transpose times the field derivative in world space equals
  // the field derivative in parametric space.
  //
  //   |                |  |        |     |       |
  //   | db0/du  db1/du |  | ds/db0 |     | ds/du |
  //   |                |  |        |     |       |
  //   | db0/dv  db1/dv |  | ds/db1 |  =  | ds/dv |
  //   |                |  |        |     |       |
  //
  // Now we just need to solve this linear system to find the derivative in
  // world space.

  bool valid;  // Ignored.
  dax::Vector2 gradient2D =
      dax::math::SolveLinearSystem(jacobianTranspose,
                                         parametricDerivative,
                                         valid);

  // We now know the gradient in our 2D space.  Convert that back to 3D space.
  dax::Vector3 gradient3D =
      space.Basis0 * gradient2D[0] + space.Basis1 * gradient2D[1];
  return gradient3D;
}


//-----------------------------------------------------------------------------
DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &daxNotUsed(parametricCoords),
    const dax::exec::CellField<dax::Vector3,dax::CellTagLine> &vertCoords,
    const dax::exec::CellField<dax::Scalar,dax::CellTagLine> &fieldValues,
    dax::CellTagLine)
{
  // The derivative of a line is in the direction of the line. Its length is
  // equal to the difference of the scalar field divided by the length of the
  // line segment. Thus, the derivative is characterized by
  // (deltaField*vec)/mag(vec)^2.

  dax::Scalar deltaField = fieldValues[1] - fieldValues[0];
  dax::Vector3 vec = vertCoords[1] - vertCoords[0];

  return (deltaField/dax::math::MagnitudeSquared(vec))*vec;
}


//-----------------------------------------------------------------------------
DAX_EXEC_EXPORT dax::Vector3 CellDerivative(
    const dax::Vector3 &daxNotUsed(parametricCoords),
    const dax::exec::CellField<dax::Vector3,dax::CellTagVertex> &daxNotUsed(vertCoords),
    const dax::exec::CellField<dax::Scalar,dax::CellTagVertex> &daxNotUsed(fieldValues),
    dax::CellTagVertex)
{
  return dax::make_Vector3(0, 0, 0);
}

}};

#endif //__dax_exec_Derivative_h
