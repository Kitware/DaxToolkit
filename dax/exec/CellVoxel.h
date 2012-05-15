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

#include <dax/exec/Field.h>

namespace dax { namespace exec {

/// All cell objects are expected to have the following methods defined:
///   Cell<type>(work);
///   GetNumberOfPoints() const;
///   GetPoint(index) const;
///   GetPoint(index, field) const;

/// A cell in a regular structured grid.
class CellVoxel
{
public:
  typedef dax::exec::internal::TopologyUniform TopologyType;

private:
  const TopologyType GridTopology;
  dax::Id CellIndex;

public:
  /// static variable that returns the number of points per cell
  const static dax::Id NUM_POINTS = 8; //needed by extract topology
  typedef dax::Tuple<dax::Id,NUM_POINTS> PointIds;

  /// Create a cell for the given work.
  DAX_EXEC_CONT_EXPORT CellVoxel(const TopologyType &gs,
                                 dax::Id index)
    : GridTopology(gs), CellIndex(index) { }

  /// Get the number of points in the cell.
  DAX_EXEC_EXPORT dax::Id GetNumberOfPoints() const
  {
    return 8;
  }

  /// Given a vertex index for a point (0 to GetNumberOfPoints() - 1), returns
  /// the index for the point in point space.
  DAX_EXEC_EXPORT dax::Id GetPointIndex(const dax::Id vertexIndex) const
  {
    dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(
          this->GetIndex(),
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

    dax::Id3 ijkPoint = ijkCell + cellVertexToPointIndex[vertexIndex];

    dax::Id pointIndex = index3ToFlatIndex(ijkPoint,
                                           this->GetGridTopology().Extent);

    return pointIndex;
  }

  /// returns the indices for all the points in the cell.
  DAX_EXEC_EXPORT PointIds GetPointIndices() const
  {
    dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(
          this->GetIndex(),
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

    PointIds pointIndices;

    pointIndices[0] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[0],
                                         this->GetGridTopology().Extent);
    pointIndices[1] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[1],
                                         this->GetGridTopology().Extent);
    pointIndices[2] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[2],
                                         this->GetGridTopology().Extent);
    pointIndices[3] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[3],
                                         this->GetGridTopology().Extent);
    pointIndices[4] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[4],
                                         this->GetGridTopology().Extent);
    pointIndices[5] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[5],
                                         this->GetGridTopology().Extent);
    pointIndices[6] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[6],
                                         this->GetGridTopology().Extent);
    pointIndices[7] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[7],
                                         this->GetGridTopology().Extent);

    return pointIndices;
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

  /// Get the cell index.  Probably only useful internally.
  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->CellIndex; }

  /// Change the cell id.  (Used internally.)
  DAX_EXEC_EXPORT void SetIndex(dax::Id cellIndex)
  {
    this->CellIndex = cellIndex;
  }

  /// Get the grid structure details.  Only useful internally.
  DAX_EXEC_EXPORT
  const TopologyType &GetGridTopology() const
  {
    return this->GridTopology;
  }
};

}}
#endif
