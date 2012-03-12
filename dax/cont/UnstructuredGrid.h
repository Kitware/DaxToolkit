/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_UnstructuredGrid_h
#define __dax_cont_UnstructuredGrid_h

#include <dax/internal/GridTopologys.h>
#include <dax/cont/ArrayHandle.h>

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

  UnstructuredGrid():
    Topology(),
    PointsCoordinates()
    {
    this->GridTopology = TopologyType();
    }

  UnstructuredGrid(const dax::cont::ArrayHandle<dax::Id>& topo,
                   const dax::cont::ArrayHandle<dax::Vector3>& coords):
    Topology(topo),
    PointsCoordinates(coords)
    {
    //I am not happy that the unstructured grid is not placed
    //in the execution enviornment on object creation
    this->GridTopology = TopologyType(this->PointsCoordinates.ReadyAsInput(),
                                      this->Topology.ReadyAsInput());
    }

  void UpdateHandles(dax::cont::ArrayHandle<dax::Id>& topo,
                     dax::cont::ArrayHandle<dax::Vector3>& coords)
    {
    if(topo.IsControlArrayValid())
      {
      topo.ReadyAsInput();
      }
    if(coords.IsControlArrayValid())
      {
      coords.ReadyAsInput();
      }

    this->Topology = topo;
    this->PointsCoordinates = coords;

    this->GridTopology = TopologyType(this->PointsCoordinates.ReadyAsOutput(),
                                      this->Topology.ReadyAsOutput());
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

    return this->PointsCoordinates.GetNumberOfEntries();
  }

  /// Get the number of cells.
  ///
  dax::Id GetNumberOfCells() const {
    return this->Topology.GetNumberOfEntries()/CellType::NUM_POINTS;
  }

  /// Gets the coordinates for a given point.
  ///
  dax::Vector3 GetPointCoordinates(dax::Id pointIndex) const {
    return this->PointsCoordinates.GetValue(pointIndex);
  }

private:
  friend class Points;
  friend class dax::cont::internal::ExecutionPackageGrid<UnstructuredGrid>;
  typedef dax::internal::TopologyUnstructured<CellType> TopologyType;

  TopologyType GridTopology;


  //control side topology
  dax::cont::ArrayHandle<dax::Id> Topology;
  dax::cont::ArrayHandle<dax::Vector3> PointsCoordinates;

};

}
}

#endif //__dax_cont_UnstructuredGrid_h
