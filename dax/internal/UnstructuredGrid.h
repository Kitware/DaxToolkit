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

  UnstructuredGrid():
    NumberOfPoints(0),
    NumberOfCells(0),
    Points(),
    Topology()
    {
    }

  UnstructuredGrid(dax::internal::DataArray<dax::Vector3>&points,
                   dax::internal::DataArray<dax::Id>& topology):
    NumberOfPoints(points.GetNumberOfEntries()),
    NumberOfCells(topology/CellType().GetNumberOfPoints()),
    Points(points),
    Topology(topology)
    {
    }

  dax::Vector3 GetPointCoordinate(dax::Id index) const
    {
    return this->Points.GetValue(index);
    }

  dax::Id GetNumberOfPoints() const { return NumberOfPoints; }
  dax::Id GetNumberOfCells() const { return NumberOfCells; }


private:
  dax::Id NumberOfPoints;
  dax::Id NumberOfCells;

  dax::internal::DataArray<dax::Vector3> Points;
  dax::internal::DataArray<dax::Id> Topology;
};

/// Returns the number of cells in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfCells(const UnstructuredGrid<T> &gridstructure)
{
  return gridstructure.GetNumberOfCells();
}

/// Returns the number of points in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const UnstructuredGrid<T> &gridstructure)
{
  return gridstructure.GetNumberOfPoints();
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const UnstructuredGrid<T> &grid,
                              dax::Id pointIndex)
{
  return grid.GetPointCoordinate(pointIndex);
}


} //internal
} //dax

#endif // __dax__internal__UnstructuredGrid_h
