#ifndef __dax__internal__UnstructuredGrid_h
#define __dax__internal__UnstructuredGrid_h

#include <dax/Types.h>
#include <dax/internal/GridStructures.h>
#include <dax/internal/DataArray.h>

namespace dax {
namespace internal {


template< typename T>
struct TopologyUnstructuredGrid
{
  typedef T CellType;

  TopologyUnstructuredGrid()
    {
    }

  TopologyUnstructuredGrid(dax::internal::DataArray<dax::Vector3>&points,
                   dax::internal::DataArray<dax::Id>& topology):    
    Points(points),
    Topology(topology)
    {
    }

  dax::internal::DataArray<dax::Vector3> Points;
  dax::internal::DataArray<dax::Id> Topology;
};

/// Returns the number of cells in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfCells(const TopologyUnstructuredGrid<T> &gridstructure)
{
  typedef typename TopologyUnstructuredGrid<T>::CellType CellType;
  return gridstructure.Topology.GetNumberOfEntries()/CellType::NUM_POINTS;
}

/// Returns the number of points in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const TopologyUnstructuredGrid<T> &gridstructure)
{
  return gridstructure.Points.GetNumberOfEntries();
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const TopologyUnstructuredGrid<T> &grid,
                              dax::Id pointIndex)
{
  return grid.Points.GetValue(pointIndex);
}


} //internal
} //dax

#endif // __dax__internal__UnstructuredGrid_h
