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
#ifndef __dax_exec_CellVoxel_h
#define __dax_exec_CellVoxel_h

#include <dax/Types.h>
#include <dax/exec/internal/TopologyUniform.h>

namespace dax { namespace exec {

/// A cell in a regular structured grid.
class CellVoxel
{
public:

  /// static variable that holds the number of points per cell
  const static dax::Id NUM_POINTS = 8;
  typedef dax::Tuple<dax::Id,NUM_POINTS> PointConnectionsType;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;

private:
  dax::exec::internal::TopologyUniform GridTopology;
  PointConnectionsType Connections;

public:
  /// Create a cell for the given work.
  DAX_CONT_EXPORT CellVoxel():
    GridTopology(),
    Connections(0)
  {
    // Suppress warnings in the copy constructor about using uninitalized
    // values. Since this constructor happens on a single thread in the control
    // environment and then copied around, the overhead is minimal.
    this->GridTopology.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
    this->GridTopology.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
    this->GridTopology.Extent.Min = dax::make_Id3(0, 0, 0);
    this->GridTopology.Extent.Max = dax::make_Id3(1, 1, 1);
  }

  /// Create a cell for the given work from a topology
  DAX_EXEC_EXPORT CellVoxel(
      const dax::exec::internal::TopologyUniform &topology,
      dax::Id cellIndex):
    GridTopology(topology),
    Connections()
  {
    dax::exec::internal::BuildCellConnectionsFromGrid(topology,cellIndex,
                                                     this->Connections);
  }

  // A COPY CONSTRUCTOR IS NEEDED TO OVERCOME THE SLOWDOWN DUE TO NVCC'S DEFAULT
  // COPY CONSTRUCTOR.
  DAX_EXEC_EXPORT CellVoxel(const CellVoxel& vox):
  GridTopology(vox.GridTopology),
  Connections(vox.Connections)
  {}

  // COPY CONSTRUCTOR (Non-Const)
  DAX_EXEC_EXPORT CellVoxel(CellVoxel& vox):
  GridTopology(vox.GridTopology),
  Connections(vox.Connections)
  {}

  /// Get the number of points in the cell.
  DAX_EXEC_EXPORT dax::Id GetNumberOfPoints() const
  {
    return NUM_POINTS;
  }

  /// Given a vertex index for a point (0 to GetNumberOfPoints() - 1), returns
  /// the index for the point in point space.
  DAX_EXEC_EXPORT dax::Id GetPointIndex(const dax::Id vertexIndex) const
  {
    return this->Connections[vertexIndex];
  }

  /// returns the indices for all the points in the cell.
  DAX_EXEC_EXPORT const PointConnectionsType& GetPointIndices() const
  {
    return this->Connections;
  }

   //  method to set this cell from a grid
  DAX_EXEC_EXPORT void BuildFromGrid(
      const dax::exec::internal::TopologyUniform &topology,
      dax::Id cellIndex)
  {
    //update our grid topology to be the same as the grids that we are now
    //based on
    this->GridTopology = topology;
    dax::exec::internal::BuildCellConnectionsFromGrid(topology,cellIndex,
                                                     this->Connections);
  }

  /// Get the origin (the location of the point at grid coordinates 0,0,0).
  DAX_EXEC_EXPORT const dax::Vector3 &GetOrigin() const
  {
    return this->GridTopology.Origin;
  }

  /// Get the spacing (the distance between grid points in each dimension).
  DAX_EXEC_EXPORT const dax::Vector3 &GetSpacing() const
  {
    return this->GridTopology.Spacing;
  }

  /// Get the extent of the grid in which this cell resides.
  DAX_EXEC_EXPORT const dax::Extent3 &GetExtent() const
  {
    return this->GridTopology.Extent;
  }

private:
  // MAKING SURE THAT THERE ARE NO MORE ASSIGNMENTS HAPPENING THAT WILL
  // POTENTIALLY BRING ABOUT A PERFOMANCE HIT
  CellVoxel & operator = (CellVoxel other);
};

}}
#endif
