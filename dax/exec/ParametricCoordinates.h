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

#ifndef __dax_exec_ParametricCoordinates_h
#define __dax_exec_ParametricCoordinates_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Derivative.h>
#include <dax/exec/Field.h>
#include <dax/exec/Interpolate.h>

#include <dax/exec/math/Matrix.h>
#include <dax/exec/math/Sign.h>
#include <dax/exec/math/VectorAnalysis.h>

#include <dax/exec/internal/FieldAccess.h>

#include <dax/exec/WorkMapCell.h>
namespace dax {
namespace exec {

//-----------------------------------------------------------------------------
/// Defines the parametric coordinates for special locations in cells.
///
template<class CellType>
struct ParametricCoordinates
#ifdef DAX_DOXYGEN_ONLY
{
  /// The location of parametric center.
  ///
  static dax::Vector3 Center();

  /// The location of each vertex.
  ///
  static dax::Tuple<dax::Vector3, CellType::NUM_POINTS> Vertex();
};
#else //DAX_DOXYGEN_ONLY
    ;
#endif

template<>
struct ParametricCoordinates<dax::exec::CellHexahedron>
{
  static dax::Vector3 Center() { return dax::make_Vector3(0.5, 0.5, 0.5); }
  static dax::Tuple<dax::Vector3, 8> Vertex() {
    const dax::Vector3 cellVertexToParametricCoords[8] = {
      dax::make_Vector3(0, 0, 0),
      dax::make_Vector3(1, 0, 0),
      dax::make_Vector3(1, 1, 0),
      dax::make_Vector3(0, 1, 0),
      dax::make_Vector3(0, 0, 1),
      dax::make_Vector3(1, 0, 1),
      dax::make_Vector3(1, 1, 1),
      dax::make_Vector3(0, 1, 1)
    };
    return dax::Tuple<dax::Vector3, 8>(cellVertexToParametricCoords);
  }
};

template<>
struct ParametricCoordinates<dax::exec::CellVoxel>
    : public ParametricCoordinates<dax::exec::CellHexahedron> {  };

template<>
struct ParametricCoordinates<dax::exec::CellTriangle>
{
  static dax::Vector3 Center() {
    return dax::make_Vector3(1.0/3.0, 1.0/3.0, 0.0);
  }
  static dax::Tuple<dax::Vector3, 3> Vertex() {
    const dax::Vector3 cellVertexToParametricCoords[3] = {
      dax::make_Vector3(0, 0, 0),
      dax::make_Vector3(1, 0, 0),
      dax::make_Vector3(0, 1, 0)
    };
    return dax::Tuple<dax::Vector3, 3>(cellVertexToParametricCoords);
  }
};

//-----------------------------------------------------------------------------
template<class WorkType, class CellType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 ParametricCoordinatesToWorldCoordinates(
    const WorkType &work,
    const CellType &cell,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    const dax::Vector3 pcoords)
{
  return dax::exec::CellInterpolate(work, cell, coordField, pcoords);
}

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 ParametricCoordinatesToWorldCoordinates(
  const WorkType &work,
  const dax::exec::CellVoxel &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 pcoords)
{
  dax::Vector3 spacing = cell.GetSpacing();
  dax::Vector3 cellOffset = spacing * pcoords;

  // This is a cheating way to get the coordinate value for index 0.  This is
  // a very special case where you would want just one point coordinate because
  // the rest are implicitly defined.
  dax::Vector3 minCoord = dax::exec::internal::FieldAccess::GetCoordinates(
        coordField, cell.GetPointIndex(0), cell.GetGridTopology(), work);

  return cellOffset + minCoord;
}

template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 WorldCoordinatesToParametricCoordinates(
  const WorkType &work,
  const dax::exec::CellVoxel &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 wcoords)
{
  // This is a cheating way to get the coordinate value for index 0.  This is
  // a very special case where you would want just one point coordinate because
  // the rest are implicitly defined.
  dax::Vector3 minCoord = dax::exec::internal::FieldAccess::GetCoordinates(
        coordField, cell.GetPointIndex(0), cell.GetGridTopology(), work);

  dax::Vector3 cellOffset = wcoords - minCoord;

  dax::Vector3 spacing = cell.GetSpacing();
  return cellOffset / spacing;
}

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 WorldCoordinatesToParametricCoordinates(
  const WorkType &work,
  const dax::exec::CellHexahedron &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 wcoords)
{
  // Use Newton's method to solve for where parametric coordinate is.
  const dax::Id MAX_ITERATIONS = 10;
  const dax::Scalar CONVERGE_DIFFERENCE = 1e-3;

  // Initial guess.
  dax::Vector3 pcoords(0.5, 0.5, 0.5);

  dax::Tuple<dax::Vector3,dax::exec::CellHexahedron::NUM_POINTS> vertexCoords =
      work.GetFieldValues(coordField);

  bool converged = false;
  for (dax::Id iteration = 0;
       !converged && (iteration < MAX_ITERATIONS);
       iteration++)
    {
    // For Newtons method, the difference in pcoords between iterations
    // is charcterized with the system of equations.
    //
    // J deltaPCoords = currentWCoords - wcoords
    //
    // The subtraction on the right side simply makes the target of the
    // solve at zero, which is what Newton's method solves for.

    dax::exec::math::Matrix3x3 jacobian
        = dax::exec::detail::make_JacobianForHexahedron(
            dax::exec::internal::derivativeWeightsHexahedron(pcoords),
            vertexCoords);

    dax::Vector3 computedWCoords =
        dax::exec::ParametricCoordinatesToWorldCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           pcoords);
    bool valid;  // Ignored.
    dax::Vector3 deltaPCoords =
        dax::exec::math::SolveLinearSystem(jacobian,
                                           computedWCoords - wcoords,
                                           valid);
    pcoords = pcoords - deltaPCoords;

    if ((dax::exec::math::Abs(deltaPCoords[0]) < CONVERGE_DIFFERENCE)
        && (dax::exec::math::Abs(deltaPCoords[1]) < CONVERGE_DIFFERENCE)
        && (dax::exec::math::Abs(deltaPCoords[2]) < CONVERGE_DIFFERENCE))
      {
      converged = true;
      }
    }

  // Not checking whether converged.
  return pcoords;
}

//-----------------------------------------------------------------------------
template<class WorkType, class ExecutionAdapter>
DAX_EXEC_EXPORT dax::Vector3 WorldCoordinatesToParametricCoordinates(
  const WorkType &work,
  const dax::exec::CellTriangle &,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 wcoords)
{
  // We will solve the world to parametric coordinates problem geometrically.
  // Consider the parallelogram formed by wcoords and p0 of the triangle and
  // the two adjacent edges. This parallelogram is equivalent to the
  // axis-aligned rectangle centered at the origin of parametric space.
  //
  //   p2 |\                 (1,0) |\                                        //
  //      | \                      |  \                                      //
  //      |  \                     |    \                                    //
  //     |    \                    |      \                                  //
  //     |     \                   |        \                                //
  //     |      \                  |    (u,v) \                              //
  //    | ---    \                 |-------*    \                            //
  //    |    ---*wcoords           |       |      \                          //
  //    |       |  \               |       |        \                        //
  // p0 *---    |   \        (0,0) *------------------\ (1,0)                //
  //        ---|     \                                                       //
  //           x--    \                                                      //
  //              ---  \                                                     //
  //                 ---\ p1                                                 //
  //
  // In this diagram, the distance between p0 and the point marked x divided by
  // the length of the edge it is on is equal, by proportionality, to the u
  // parametric coordiante. (The v coordinate follows the other edge
  // accordingly.) Thus, if we can find the interesection at x (or more
  // specifically the distance between p0 and x), then we can find that
  // parametric coordinate.
  //
  // Because the triangle is in 3-space, we are actually going to intersect the
  // edge with a plane that is parallel to the opposite edge of p0 and
  // perpendicular to the triangle. This is partially because it is easy to
  // find the intersection between a plane and a line and partially because the
  // computation will work for points not on the plane. (The result is
  // equivalent to a point projected on the plane.)
  //
  // First, we define an implicit plane as:
  //
  // dot((p - wcoords), planeNormal) = 0
  //
  // where planeNormal is the normal to the plane (easily computed from the
  // triangle), and p is any point in the plane. Next, we define the parametric
  // form of the line:
  //
  // p(d) = (p1 - p0)d + p0
  //
  // Where d is the fraction of distance from p0 toward p1. Note that d is
  // actually equal to the parametric coordinate we are trying to find. Once we
  // compute it, we are done. We can skip the part about finding the actual
  // coordinates of the intersection.
  //
  // Solving for the interesection is as simple as substituting the line's
  // definition of p(d) into p for the plane equation. With some basic algebra
  // you get:
  //
  // d = dot((wcoords - p0), planeNormal)/dot((p1-p0), planeNormal)
  //
  // From here, the u coordiante is simply d/mag(p1-p0).  The v coordinate
  // follows similarly.
  //

  dax::Vector3 pcoords(dax::Scalar(0));
  dax::Tuple<dax::Vector3, 3> vertexCoords = work.GetFieldValues(coordField);
  dax::Vector3 triangleNormal =
      dax::exec::math::TriangleNormal(vertexCoords[0],
                                      vertexCoords[1],
                                      vertexCoords[2]);

  for (int dimension = 0; dimension < 2; dimension++)
    {
    const dax::Vector3 &p0 = vertexCoords[0];
    const dax::Vector3 &p1 = vertexCoords[dimension+1];
    const dax::Vector3 &p2 = vertexCoords[2-dimension];
    dax::Vector3 planeNormal = dax::exec::math::Cross(triangleNormal, p2-p0);

    dax::Scalar d =
        dax::dot(wcoords - p0, planeNormal)/dax::dot(p1 - p0, planeNormal);

    pcoords[dimension] = d;
    }

  return pcoords;
}

}
}

#endif //__dax_exec_ParametricCoordinates_h
