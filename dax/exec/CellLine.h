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
  DAX_EXEC_EXPORT CellLine()
    :Connections()
    { }

  /// Create a cell for the given work from a topology
  template<class ConnectionsPortalT>
  DAX_EXEC_EXPORT CellLine(
      const dax::exec::internal::TopologyUnstructured<
        CellLine,ConnectionsPortalT> &topology,
      dax::Id cellIndex)
  {
    dax::exec::internal::BuildCellConnectionsFromGrid(topology,cellIndex,
                                           this->Connections);
  }

  // A COPY CONSTRUCTOR IS NEEDED TO OVERCOME THE SLOWDOWN DUE TO NVCC'S DEFAULT
  // COPY CONSTRUCTOR.
  DAX_EXEC_EXPORT CellLine(const CellLine& line)
  :Connections(line.Connections)
  {}

  // COPY CONSTRUCTOR (Non-Const)
  DAX_EXEC_EXPORT CellLine(CellLine& line)
  :Connections(line.Connections)
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

  // method to set this cell from a portal
  template<class ConnectionsPortalT>
  DAX_EXEC_EXPORT void BuildFromGrid(
   const dax::exec::internal::TopologyUnstructured<
    CellLine,ConnectionsPortalT> &topology,
    dax::Id cellIndex)
  {
    dax::exec::internal::BuildCellConnectionsFromGrid(topology,cellIndex,
                                           this->Connections);
  }

  //  method to set this cell from a different tuple
  DAX_EXEC_EXPORT void SetPointIndices(
      const PointConnectionsType & cellConnections)
  {
    this->Connections = cellConnections;
  }

private:
  // MAKING SURE THAT THERE ARE NO MORE ASSIGNMENTS HAPPENING THAT WILL
  // POTENTIALLY BRING ABOUT A PERFOMANCE HIT
  CellLine & operator = (CellLine other);

};

}}
#endif // __dax_exec_CellLine_h
