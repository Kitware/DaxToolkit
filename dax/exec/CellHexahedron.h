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
#ifndef __dax_exec_CellHexahedron_h
#define __dax_exec_CellHexahedron_h

#include <dax/Types.h>
#include <dax/internal/GridTopologys.h>

#include <dax/exec/Field.h>

namespace dax { namespace exec {

/// All cell objects are expected to have the following methods defined:
///   Cell<type>(work);
///   GetNumberOfPoints() const;
///   GetPoint(index) const;
///   GetPoint(index, field) const;

/// A cell in a regular structured grid.
class CellHexahedron
{
public:
  typedef dax::internal::TopologyUnstructured<CellHexahedron> TopologyType;

private:
  const TopologyType GridTopology;
  dax::Id CellIndex;
  dax::Id TopologyPosition;
public:
  /// static variable that returns the number of points per cell
  const static dax::Id NUM_POINTS = 8; //needed by unstructured grid
  /// Create a cell for the given work.
  DAX_EXEC_CONT_EXPORT CellHexahedron(
      const TopologyType  &gs,
      dax::Id index)
    : GridTopology(gs),
      CellIndex(index),
      TopologyPosition(NUM_POINTS*index)
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
    return this->GridTopology.Topology.GetValue(
          this->TopologyPosition+vertexIndex);
  }

  /// Get the cell index.  Probably only useful internally.
  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->CellIndex; }

  /// Change the cell id.  (Used internally.)
  DAX_EXEC_EXPORT void SetIndex(dax::Id cellIndex)
  {
    this->CellIndex = cellIndex;
    this->TopologyPosition = NUM_POINTS*cellIndex;
  }

  /// Get the grid structure details.  Only useful internally.
  DAX_EXEC_EXPORT
  const TopologyType& GetGridTopology() const
  {
    return this->GridTopology;
  }
};

}}
#endif
