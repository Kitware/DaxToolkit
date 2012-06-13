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
#ifndef __dax__exec__internal__TopologyUnstructured_h
#define __dax__exec__internal__TopologyUnstructured_h

#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {


/// The basic data describing the topology of an unstructured grid. It
/// comprises two arrays: an array of point coordinates and an array of cell
/// connections. The unstructured grid is typed on the cell type. Only that one
/// cell type can be represented in the unstructured grid at any one time.
/// Also, the data in the arrays is meant to be immutable. To generate new
/// topology, you have to create these two arrays outside of this structure and
/// then bring them together. This prevents the structure from getting into
/// invalid states (as well as get around problems with const vs. non-const
/// arrays).
///
template<typename T, class ExecutionAdapter>
struct TopologyUnstructured
{
  typedef T CellType;
  typedef typename ExecutionAdapter::template FieldStructures<dax::Id>
      ::IteratorConstType CellConnectionsIteratorType;

  TopologyUnstructured()
    : CellConnections(CellConnectionsIteratorType()),
      NumberOfPoints(0),
      NumberOfCells(0)
    {
    }

  /// Create a topology with the given descriptive arrays.
  ///
  /// \param numberOfPoints The number of points in the grid.
  /// \param cellConnections An array containing a list for each cell giving
  /// the point index for each vertex of the cell.  The length of this array
  /// should be \c numberOfCells times \c CellType::NUM_POINTS.
  /// \param numberOfCells The number of cells in the grid.
  ///
  TopologyUnstructured(CellConnectionsIteratorType cellConnections,
                       dax::Id numberOfPoints,
                       dax::Id numberOfCells)
    : CellConnections(cellConnections),
      NumberOfPoints(numberOfPoints), NumberOfCells(numberOfCells)
  {
  }

  CellConnectionsIteratorType CellConnections;
  dax::Id NumberOfPoints;
  dax::Id NumberOfCells;
};

/// Returns the number of cells in a unstructured grid.
///
template<typename T, class ExecutionAdapter>
DAX_EXEC_EXPORT
dax::Id numberOfCells(const TopologyUnstructured<T, ExecutionAdapter> &topology)
{
  return topology.NumberOfCells;
}

/// Returns the number of points in a unstructured grid.
///
template<typename T, class ExecutionAdapter>
DAX_EXEC_EXPORT
dax::Id numberOfPoints(const TopologyUnstructured<T,ExecutionAdapter> &topology)
{
  return topology.NumberOfPoints;
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
///
template<typename T, class ExecutionAdapter>
DAX_EXEC_EXPORT
dax::Vector3 pointCoordiantes(
    const TopologyUnstructured<T, ExecutionAdapter> &topology,
    dax::Id pointIndex)
{
  return *(topology.PointCoordinates + pointIndex);
}

} //internal
} //exec
} //dax

#endif // __dax__exec__internal__TopologyUnstructured_h
