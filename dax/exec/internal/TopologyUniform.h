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
#ifndef __dax__exec__internal__TopologyUniform_h
#define __dax__exec__internal__TopologyUniform_h

#include <dax/CellTag.h>
#include <dax/Extent.h>

namespace dax {
namespace exec {
namespace internal {

/// Contains all the parameters necessary to specify the topology of a uniform
/// rectilinear grid.
///
struct TopologyUniform {
  typedef dax::CellTagVoxel CellTag;

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;

  /// Returns the number of points in a uniform rectilinear grid.
  ///
  DAX_EXEC_EXPORT
  dax::Id GetNumberOfPoints() const
  {
    dax::Id3 dims = dax::extentDimensions(this->Extent);
    return dims[0]*dims[1]*dims[2];
  }

  /// Returns the number of cells in a uniform rectilinear grid.
  ///
  DAX_EXEC_EXPORT
  dax::Id GetNumberOfCells() const
  {
    dax::Id3 dims = dax::extentDimensions(this->Extent)
                    - dax::make_Id3(1, 1, 1);
    return dims[0]*dims[1]*dims[2];
  }

  /// Returns the point position in a structured grid for a given i, j, and k
  /// value stored in /c ijk
  ///
  DAX_EXEC_EXPORT
  dax::Vector3 GetPointCoordiantes(dax::Id3 ijk) const
  {
    return dax::make_Vector3(this->Origin[0] + ijk[0] * this->Spacing[0],
                             this->Origin[1] + ijk[1] * this->Spacing[1],
                             this->Origin[2] + ijk[2] * this->Spacing[2]);
  }

  /// Returns the point position in a structured grid for a given index
  /// which is represented by /c pointIndex
  ///
  DAX_EXEC_EXPORT
  dax::Vector3 GetPointCoordiantes(dax::Id pointIndex) const
  {
    dax::Id3 ijk = flatIndexToIndex3(pointIndex, this->Extent);
    return this->GetPointCoordiantes(ijk);
  }
} __attribute__ ((aligned(4)));

template<class PointConnectionType>
DAX_EXEC_EXPORT
void BuildCellConnectionsFromGrid(
        const dax::exec::internal::TopologyUniform& grid,
        dax::Id cellIndex,
        PointConnectionType& connections)
{
  dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(
          cellIndex,
          grid.Extent);

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

    connections[0] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[0],
                                         grid.Extent);
    connections[1] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[1],
                                         grid.Extent);
    connections[2] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[2],
                                         grid.Extent);
    connections[3] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[3],
                                         grid.Extent);
    connections[4] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[4],
                                         grid.Extent);
    connections[5] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[5],
                                         grid.Extent);
    connections[6] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[6],
                                         grid.Extent);
    connections[7] = index3ToFlatIndex(ijkCell + cellVertexToPointIndex[7],
                                         grid.Extent);
}

}  }  } //namespace dax::exec::internal

#endif //__dax__exec__internal__TopologyUniform_h
