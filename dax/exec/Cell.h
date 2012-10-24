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
#ifndef __dax_exec_Cell_h
#define __dax_exec_Cell_h

#include <dax/exec/CellHexahedron.h>
#include <dax/exec/CellLine.h>
#include <dax/exec/CellQuadrilateral.h>
#include <dax/exec/CellTetrahedron.h>
#include <dax/exec/CellTriangle.h>
#include <dax/exec/CellVertex.h>
#include <dax/exec/CellVoxel.h>
#include <dax/exec/CellWedge.h>

#ifdef DAX_DOXYGEN_ONLY

namespace dax {
namespace exec {

/// \brief Cell descriptor class.
///
/// A Cell class provides a description of a cell. This class does not actually
/// exist. Rather, this documentation is provided to describe the interface all
/// Cell objects should provide.
///
class Cell
{
public:
  /// Static variable that contains the number of points per cell.
  ///
  const static dax::Id NUM_POINTS = 4;

  /// A typedef of a dax::Tuple that can hold the point connections for this
  /// cell.  This defines the topology of the cell.
  ///
  typedef dax::Tuple<dax::Id, NUM_POINTS> PointConnectionsType;

  /// This defines the topological dimensions of the class. 3 for polyhedra,
  /// 2 for polygons, 1 for lines, 0 for points.
  ///
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;

  /// Cell classes are created internally within Dax, but the constructor
  /// generally takes on this form.
  ///
  DAX_EXEC_EXPORT Cell();

  /// Returns the number of points in the cell. Basically regurgitates
  /// NUM_POINTS.
  ///
  DAX_EXEC_EXPORT dax::Id GetNumberOfPoints() const;

  /// Given a vertex index for a point (0 to NUM_POINTS-1), returns the index
  /// for the point in point space.
  ///
  DAX_EXEC_EXPORT dax::Id GetPointIndex(dax::Id vertexIndex) const;

  /// Returns the indices for all the vertex connections in the cell.
  ///
  DAX_EXEC_EXPORT const PointConnectionsType& GetPointIndices() const;

  /// Specifify the indices that this cell represents given a grid and cell index
  template<class GridType>
  DAX_EXEC_EXPORT void SetPointIndices(const GridType &topology, dax::Id cellIndex);

  /// Specifiy the indices of this cell given an other PointConnectionsType object
  /// Only works on cell types that arent implicit.
  DAX_EXEC_EXPORT void SetPointIndices(const PointConnectionsType &cellConnections);
};

}
} // namespace dax::exec

#endif // DAX_DOXYGEN_ONLY

#endif // __dax_exec_Cell_h
