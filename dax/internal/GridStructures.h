/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax__internal__GridStructures_h
#define __dax__internal__GridStructures_h

#include <dax/Types.h>

namespace dax {
namespace internal {

/// Extent3 stores the 6 values for the extents of a structured grid array.
/// It gives the minimum indices and the maximum indices.
struct Extent3 {
  Int3 Min;
  Int3 Max;
} __attribute__ ((aligned(4)));

/// Given an extent, returns the array dimensions in each direction.
inline dax::Int3 extentDimensions(const Extent3 &extent)
{
  return extent.Max - extent.Min + make_Int3(1, 1, 1);
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts a flat index to the r,s,t
/// 3d indices.
inline dax::Int3 flatIndexToInt3Index(dax::Id index, const Extent3 &extent)
{
  dax::Int3 dims = extentDimensions(extent);

  dax::Int3 ijk;
  ijk.x = index % (dims.x - 1);
  ijk.y = (index / (dims.x - 1)) % (dims.y -1 );
  ijk.z = (index / ((dims.x - 1) * (dims.y -1 )));

  return ijk + extent.Min;
}

/// Same as flatIndexToInt3Index except performed for cells using extents for
/// for points, which have one more in every direction than cells.
inline dax::Int3 flatIndexToInt3IndexCell(dax::Id index,
                                          const Extent3 &pointExtent)
{
  dax::Int3 dims = extentDimensions(pointExtent) - dax::make_Int3(1, 1, 1);

  dax::Int3 ijk;
  ijk.x = index % (dims.x - 1);
  ijk.y = (index / (dims.x - 1)) % (dims.y -1 );
  ijk.z = (index / ((dims.x - 1) * (dims.y -1 )));

  return ijk + pointExtent.Min;
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts r,s,t 3d indices to a flat
/// index.
inline dax::Id int3IndexToFlatIndex(dax::Int3 ijk, const Extent3 &extent)
{
  dax::Int3 dims = extentDimensions(extent);
  dax::Int3 deltas = ijk - extent.Min;

  return deltas.x + dims.x*(deltas.y + dims.y*deltas.z);
}

/// Same as int3IndexToFlatIndex except performed for cells using extents for
/// for points, which have one more in every direction than cells.
inline dax::Id int3IndexToFlatIndexCell(dax::Int3 ijk,
                                        const Extent3 &pointExtent)
{
  dax::Int3 dims = extentDimensions(pointExtent) - dax::make_Int3(1, 1, 1);
  dax::Int3 deltas = ijk - pointExtent.Min;

  return deltas.x + dims.x*(deltas.y + dims.y*deltas.z);
}

struct StructureUniformGrid {
  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
} __attribute__ ((aligned(4)));

}  }

#endif //__dax__internal__GridStructures_h
