  /*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax__internal__GridStructures_h
#define __dax__internal__GridStructures_h

#include <dax/Extent.h>

namespace dax {
namespace internal {

struct StructureUniformGrid {
  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
} __attribute__ ((aligned(4)));

/// Returns the number of points in a structured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const T &gridstructure)
{
  dax::Id3 dims = dax::extentDimensions(gridstructure.Extent);
  return dims[0]*dims[1]*dims[2];
}

template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfCells(const T &gridstructure)
{
  dax::Id3 dims = dax::extentDimensions(gridstructure.Extent)
                  - dax::make_Id3(1, 1, 1);
  return dims[0]*dims[1]*dims[2];
}

}  }

#endif //__dax__internal__GridStructures_h
