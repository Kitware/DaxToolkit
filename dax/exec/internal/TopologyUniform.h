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

#include <dax/Extent.h>

namespace dax {
namespace exec {
class CellVoxel;
}
}

namespace dax {
namespace exec {
namespace internal {

/// Contains all the parameters necessary to specify the topology of a uniform
/// rectilinear grid.
///
struct TopologyUniform {
  typedef dax::exec::CellVoxel CellType;

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
} __attribute__ ((aligned(4)));

/// Returns the number of points in a uniform rectilinear grid.
///
DAX_EXEC_EXPORT
dax::Id numberOfPoints(const TopologyUniform &GridTopology)
{
  dax::Id3 dims = dax::extentDimensions(GridTopology.Extent);
  return dims[0]*dims[1]*dims[2];
}

/// Returns the number of cells in a uniform rectilinear grid.
///
template<typename T>
DAX_EXEC_EXPORT
dax::Id numberOfCells(const T &GridTopology)
{
  dax::Id3 dims = dax::extentDimensions(GridTopology.Extent)
                  - dax::make_Id3(1, 1, 1);
  return dims[0]*dims[1]*dims[2];
}

/// Returns the point position in a structured grid for a given i, j, and k
/// value stored in /c ijk
///
DAX_EXEC_EXPORT
dax::Vector3 pointCoordiantes(const TopologyUniform &grid,
                              dax::Id3 ijk)
{
  dax::Vector3 origin = grid.Origin;
  dax::Vector3 spacing = grid.Spacing;
  return dax::make_Vector3(origin[0] + ijk[0] * spacing[0],
                           origin[1] + ijk[1] * spacing[1],
                           origin[2] + ijk[2] * spacing[2]);
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
///
DAX_EXEC_EXPORT
dax::Vector3 pointCoordiantes(const TopologyUniform &grid,
                              dax::Id pointIndex)
{
  dax::Id3 ijk = flatIndexToIndex3(pointIndex, grid.Extent);
  return pointCoordiantes(grid, ijk);
}

template<class PointConnectionType>
DAX_EXEC_EXPORT
void BuildCellConnectionsFromGrid(
        const dax::exec::internal::TopologyUniform& grid,
        dax::Id cellIndex,
        PointConnectionType& connections)
{
  dax::Id3 dimensions = dax::extentDimensions(grid.Extent);

  dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(
          cellIndex,
          grid.Extent);

  connections[0] = index3ToFlatIndex(ijkCell, grid.Extent);
  connections[1] = connections[0] + 1;
  connections[2] = connections[0] + dimensions[0] + 1;
  connections[3] = connections[0] + dimensions[0];

  connections[4] = connections[0] + dimensions[0]*dimensions[1];
  connections[5] = connections[1] + dimensions[0]*dimensions[1];
  connections[6] = connections[2] + dimensions[0]*dimensions[1];
  connections[7] = connections[3] + dimensions[0]*dimensions[1];
}

}  }  } //namespace dax::exec::internal

#endif //__dax__exec__internal__TopologyUniform_h
