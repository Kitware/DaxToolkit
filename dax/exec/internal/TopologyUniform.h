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

#include <dax/exec/CellVertices.h>

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

  /// Returns the point indices for all vertices.
  ///
  DAX_EXEC_EXPORT
  dax::exec::CellVertices<CellTag> GetCellConnections(dax::Id cellIndex) const
  {
    dax::Id3 dimensions = dax::extentDimensions(this->Extent);

    dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(cellIndex, this->Extent);

    dax::exec::CellVertices<CellTag> connections;
    connections[0] = index3ToFlatIndex(ijkCell, this->Extent);
    connections[1] = connections[0] + 1;
    connections[2] = connections[0] + dimensions[0] + 1;
    connections[3] = connections[0] + dimensions[0];

    connections[4] = connections[0] + dimensions[0]*dimensions[1];
    connections[5] = connections[1] + dimensions[0]*dimensions[1];
    connections[6] = connections[2] + dimensions[0]*dimensions[1];
    connections[7] = connections[3] + dimensions[0]*dimensions[1];

    return connections;
  }
} __attribute__ ((aligned(4)));

}  }  } //namespace dax::exec::internal

#endif //__dax__exec__internal__TopologyUniform_h
