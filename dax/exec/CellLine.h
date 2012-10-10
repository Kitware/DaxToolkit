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
#ifndef __dax_exec_CellLine_h
#define __dax_exec_CellLine_h


#include <dax/Types.h>
#include <dax/exec/internal/TopologyUnstructured.h>

namespace dax { namespace exec {

class CellLine
{
public:

  /// static variable that returns the number of points per cell
  const static dax::Id NUM_POINTS = 2;
  typedef dax::Tuple<dax::Id,NUM_POINTS> PointConnectionsType;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 1;

private:
  PointConnectionsType Connections;

public:
  /// Create a cell for the given work.
  template<class ExecutionAdapter>
  DAX_EXEC_EXPORT CellLine(
    const dax::exec::internal::TopologyUnstructured<
      CellLine,ExecutionAdapter> &topology)
    :Connections(GetPointConnections()
    { }

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
  DAX_EXEC_EXPORT PointConnectionsType GetPointIndices() const
  {
    return this->Connections;
  }

  // method to set this cell from a portal
  template<class PortalType>
  DAX_EXEC_EXPORT static void SetPointIndicies(
      const PortalType & cellConnectionsPortal,
      dax::Id cellIndex)
  {
    dax::Id offset = cellIndex*NUM_POINTS;
    this->Connection[0] = cellConnectionsPortal.Get(offset + 0);
    this->Connection[1] = cellConnectionsPortal.Get(offset + 1);
  }

  //  method to set this cell from a different tuple
  DAX_EXEC_EXPORT static void SetPointIndicies(
      const PointConnectionsType & cellConnections)
  {
    this->Connections = cellConnections;
  }
};

}}
#endif // __dax_exec_CellLine_h
