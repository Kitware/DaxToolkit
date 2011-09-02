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
  Id3 Min;
  Id3 Max;
} __attribute__ ((aligned(4)));

/// Given an extent, returns the array dimensions in each direction.
inline dax::Id3 extentDimensions(const Extent3 &extent)
{
  return extent.Max - extent.Min + make_Id3(1, 1, 1);
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts a flat index to the r,s,t
/// 3d indices.
inline dax::Id3 flatIndexToIndex3(dax::Id index, const Extent3 &extent)
{
  dax::Id3 dims = extentDimensions(extent);

  dax::Id3 ijk;
  ijk.x = index % dims.x;
  ijk.y = (index / dims.x) % dims.y;
  ijk.z = (index / (dims.x * dims.y));

  return ijk + extent.Min;
}

/// Same as flatIndexToIndex3 except performed for cells using extents for
/// for points, which have one more in every direction than cells.
inline dax::Id3 flatIndexToIndex3Cell(dax::Id index,
                                          const Extent3 &pointExtent)
{
  dax::Id3 dims = extentDimensions(pointExtent) - dax::make_Id3(1, 1, 1);

  dax::Id3 ijk;
  ijk.x = index % dims.x;
  ijk.y = (index / dims.x) % dims.y;
  ijk.z = (index / (dims.x * dims.y));

  return ijk + pointExtent.Min;
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts r,s,t 3d indices to a flat
/// index.
inline dax::Id index3ToFlatIndex(dax::Id3 ijk, const Extent3 &extent)
{
  dax::Id3 dims = extentDimensions(extent);
  dax::Id3 deltas = ijk - extent.Min;

  return deltas.x + dims.x*(deltas.y + dims.y*deltas.z);
}

/// Same as index3ToFlatIndex except performed for cells using extents for
/// for points, which have one more in every direction than cells.
inline dax::Id index3ToFlatIndexCell(dax::Id3 ijk,
                                        const Extent3 &pointExtent)
{
  dax::Id3 dims = extentDimensions(pointExtent) - dax::make_Id3(1, 1, 1);
  dax::Id3 deltas = ijk - pointExtent.Min;

  return deltas.x + dims.x*(deltas.y + dims.y*deltas.z);
}

struct StructureUniformGrid {
  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;
} __attribute__ ((aligned(4)));

}  }

#endif //__dax__internal__GridStructures_h
