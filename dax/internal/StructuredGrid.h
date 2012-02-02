/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax__internal__StructuredGrid_h
#define __dax__internal__StructuredGrid_h

#include <dax/Extent.h>

namespace dax {
namespace internal {

struct StructureUniformGrid {
  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
} __attribute__ ((aligned(4)));

/// Returns the number of points in a structured grid.
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const StructureUniformGrid &gridstructure)
{
  dax::Id3 dims = dax::extentDimensions(gridstructure.Extent);
  return dims[0]*dims[1]*dims[2];
}

/// Returns the number of cells in a structured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfCells(const T &gridstructure)
{
  dax::Id3 dims = dax::extentDimensions(gridstructure.Extent)
                  - dax::make_Id3(1, 1, 1);
  return dims[0]*dims[1]*dims[2];
}

/// Returns the point position in a structured grid for a given i, j, and k
/// value stored in /c ijk
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const StructureUniformGrid &grid,
                              dax::Id3 ijk)
{
  dax::Vector3 origin = grid.Origin;
  dax::Vector3 spacing = grid.Spacing;
  return dax::make_Vector3(origin[0] + ijk[0] * spacing[0],
                           origin[1] + ijk[1] * spacing[1],
                           origin[2] + ijk[2] * spacing[2]);
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const StructureUniformGrid &grid,
                              dax::Id pointIndex)
{
  dax::Id3 ijk = flatIndexToIndex3(pointIndex, grid.Extent);
  return pointCoordiantes(grid, ijk);
}

}  }

#endif //__dax__internal__StructuredGrid_h
