#ifndef __dax__internal__UnstructuredGrid_h
#define __dax__internal__UnstructuredGrid_h

#include <dax/Types.h>
#include <dax/internal/GridStructures.h>
#include <dax/internal/DataArray.h>

namespace dax {
namespace internal {


template< typename T>
struct UnstructuredGrid
{
  typedef T CellType;

  UnstructuredGrid()
    {
    }

  UnstructuredGrid(dax::internal::DataArray<dax::Vector3>&points,
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
dax::Id numberOfCells(const UnstructuredGrid<T> &gridstructure)
{
  typedef typename UnstructuredGrid<T>::CellType CellType;
  return gridstructure.Topology.GetNumberOfEntries()/CellType::NUM_POINTS;
}

/// Returns the number of points in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const UnstructuredGrid<T> &gridstructure)
{
  return gridstructure.Points.GetNumberOfEntries();
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const UnstructuredGrid<T> &grid,
                              dax::Id pointIndex)
{
  return grid.Points.GetValue(pointIndex);
}


} //internal
} //dax

#endif // __dax__internal__UnstructuredGrid_h
