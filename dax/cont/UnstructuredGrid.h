/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_UnstructuredGrid_h
#define __dax_cont_UnstructuredGrid_h

#include <dax/internal/GridTopologys.h>

namespace dax { namespace cont { namespace internal {
template<class Grid> class ExecutionPackageGrid;
} } }

namespace dax { namespace exec {
class CellHexahedron;
} }

namespace dax {
namespace cont {

/// This class defines the topology of an unstructured grid. An unstructured
/// grid can only contain cells of a single type.
///
template <typename CellT>
class UnstructuredGrid
{
public:
  typedef CellT CellType;

  UnstructuredGrid()
    {
    }
  /// A simple class representing the points in an unstructured grid.
  ///
  class Points
  {
  public:
    Points(const UnstructuredGrid &grid) : GridTopology(grid.GridTopology) { }
    dax::Vector3 GetCoordinates(dax::Id pointIndex) const {
      return dax::internal::pointCoordiantes(this->GridTopology, pointIndex);
    }
    const dax::internal::TopologyUniform &GetStructureForExecution() const {
      return this->GridTopology;
    }
  private:
    dax::internal::TopologyUnstructured<CellType> GridTopology;
  };

  /// Returns an object representing the points in a uniform grid. Most helpful
  /// in passing point fields to worklets.
  ///
  Points GetPoints() const { return Points(*this); }

  // Helper functions

  /// Get the number of points.
  ///
  dax::Id GetNumberOfPoints() const {
    return dax::internal::numberOfPoints(this->GridTopology);
  }

  /// Get the number of cells.
  ///
  dax::Id GetNumberOfCells() const {
    return dax::internal::numberOfCells(this->GridTopology);
  }

  /// Gets the coordinates for a given point.
  ///
  dax::Vector3 GetPointCoordinates(dax::Id pointIndex) const {
    return dax::internal::pointCoordiantes(this->GridTopology, pointIndex);
  }

private:
  friend class Points;
  friend class dax::cont::internal::ExecutionPackageGrid<UnstructuredGrid>;

  typedef dax::internal::TopologyUnstructured<CellType> TopologyType;
  TopologyType GridTopology;

};

}
}

#endif //__dax_cont_UnstructuredGrid_h
