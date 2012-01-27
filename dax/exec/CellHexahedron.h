/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_CellHexahedron_h
#define __dax_exec_CellHexahedron_h

#include <dax/Types.h>
#include <dax/internal/GridStructures.h>

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
private:
  const dax::internal::UnstructuredGrid<CellHexahedron> GridStructure;
  dax::Id CellIndex;
  dax::Id TopologyPosition;

public:
  /// Create a cell for the given work.
  DAX_EXEC_CONT_EXPORT CellHexahedron(
      const dax::internal::UnstructuredGrid<CellHexahedron>  &gs,
      dax::Id index)
    : GridStructure(gs),
      CellIndex(index),
      TopologyPosition(8*index)
    { }

  /// Get the number of points in the cell.
  DAX_EXEC_EXPORT dax::Id GetNumberOfPoints() const
  {
    return 8;
  }

  /// Given a vertex index for a point (0 to GetNumberOfPoints() - 1), returns
  /// the index for the point in point space.
  DAX_EXEC_EXPORT dax::Id GetPointIndex(const dax::Id vertexIndex) const
  {    
    return this->TopologyPosition+vertexIndex;
  }

  /// Get the cell index.  Probably only useful internally.
  DAX_EXEC_EXPORT dax::Id GetIndex() const { return this->CellIndex; }

  /// Change the cell id.  (Used internally.)
  DAX_EXEC_EXPORT void SetIndex(dax::Id cellIndex)
  {
    this->CellIndex = cellIndex;
    this->TopologyPosition = 8*cellIndex;
  }


  /// Get the grid structure details.  Only useful internally.
  DAX_EXEC_EXPORT
  const dax::internal::UnstructuredGrid<CellHexahedron>& GetGridStructure() const
  {
    return this->GridStructure;
  }
};

}}
#endif
