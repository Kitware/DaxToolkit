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

    dax::exec::math::Matrix<dax::Scalar,3,3> jacobian
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
        || (dax::exec::math::Abs(deltaPCoords[1]) < CONVERGE_DIFFERENCE)
        || (dax::exec::math::Abs(deltaPCoords[2]) < CONVERGE_DIFFERENCE))
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
  const dax::exec::CellTriangle &cell,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
  const dax::Vector3 wcoords)
{
  // Not implemented yet.
  return dax::make_Vector3(-1,-1,-1);
}

}
}

#endif //__dax_exec_ParametricCoordinates_h
