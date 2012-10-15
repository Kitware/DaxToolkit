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
  const dax::exec::internal::TopologyUniform GridTopology;
  PointConnectionsType Connections;

public:
  /// Create a cell for the given work.
  DAX_EXEC_EXPORT explicit CellVoxel(const dax::exec::internal::TopologyUniform &gs)
    : GridTopology(gs), Connections()
    {

    }

  /// Get the number of points in the cell.
  DAX_EXEC_EXPORT dax::Id GetNumberOfPoints() const
  {
    return NUM_POINTS;
  }

  /// Given a vertex index for a point (0 to GetNumberOfPoints() - 1), returns
  /// the index for the point in point space.
  DAX_EXEC_EXPORT dax::Id GetPointIndex(const dax::Id vertexIndex) const
  {
    return this->GetPointIndices()[vertexIndex];
  }

  /// returns the indices for all the points in the cell.
  DAX_EXEC_EXPORT const PointConnectionsType& GetPointIndices() const
  {
    return this->Connections;
  }

   //  method to set this cell from a portal
  template<class PortalType>
  DAX_EXEC_EXPORT void SetPointIndices(
       const PortalType &,
      dax::Id cellIndex)
  {
    dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(
          cellIndex,
          this->GetGridTopology().Extent);

    const dax::Id3 cellVertexToPointIndex[8] = {
      dax::make_Id3(0, 0, 0),
      dax::make_Id3(1, 0, 0),
      dax::make_Id3(1, 1, 0),
      dax::make_Id3(0, 1, 0),
      dax::make_Id3(0, 0, 1),
      dax::make_Id3(1, 0, 1),
      dax::make_Id3(1, 1, 1),
      dax::make_Id3(0, 1, 1)
    };

    this->Connections[0] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[0],
                                         this->GetGridTopology().Extent);
    this->Connections[1] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[1],
                                         this->GetGridTopology().Extent);
    this->Connections[2] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[2],
                                         this->GetGridTopology().Extent);
    this->Connections[3] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[3],
                                         this->GetGridTopology().Extent);
    this->Connections[4] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[4],
                                         this->GetGridTopology().Extent);
    this->Connections[5] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[5],
                                         this->GetGridTopology().Extent);
    this->Connections[6] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[6],
                                         this->GetGridTopology().Extent);
    this->Connections[7] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[7],
                                         this->GetGridTopology().Extent);
  }

  /// Get the origin (the location of the point at grid coordinates 0,0,0).
  DAX_EXEC_EXPORT const dax::Vector3 &GetOrigin() const
  {
    return this->GetGridTopology().Origin;
  }

  /// Get the spacing (the distance between grid points in each dimension).
  DAX_EXEC_EXPORT const dax::Vector3 &GetSpacing() const
  {
    return this->GetGridTopology().Spacing;
  }

  /// Get the extent of the grid in which this cell resides.
  DAX_EXEC_EXPORT const dax::Extent3 &GetExtent() const
  {
    return this->GetGridTopology().Extent;
  }

  /// Get the grid structure details.  Only useful internally.
  DAX_EXEC_EXPORT
  const dax::exec::internal::TopologyUniform &GetGridTopology() const
  {
    return this->GridTopology;
  }

  // A COPY CONSTRUCTOR IS NEEDED TO OVERCOME THE SLOWDOWN DUE TO NVCC'S DEFAULT
  // COPY CONSTRUCTOR.
  DAX_EXEC_EXPORT CellVoxel(const CellVoxel& vox):
  GridTopology(vox.GridTopology),
  Connections(vox.Connections)
  {}

private:
  // MAKING SURE THAT THERE ARE NO MORE ASSIGNMENTS HAPPENING THAT WILL
  // POTENTIALLY BRING ABOUT A PERFOMANCE HIT
  CellVoxel & operator = (CellVoxel other);
};

}}
#endif
