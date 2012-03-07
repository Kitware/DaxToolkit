//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax__Extent_h
#define __dax__Extent_h

#include <dax/Types.h>

namespace dax {

/// Extent3 stores the 6 values for the extents of a structured grid array.
/// It gives the minimum indices and the maximum indices.
struct Extent3 {
  Id3 Min;
  Id3 Max;
} __attribute__ ((aligned(4)));

/// Given an extent, returns the array dimensions in each direction.
DAX_EXEC_CONT_EXPORT dax::Id3 extentDimensions(const Extent3 &extent)
{
  return extent.Max - extent.Min + make_Id3(1, 1, 1);
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts a flat index to the r,s,t
/// 3d indices.
DAX_EXEC_CONT_EXPORT dax::Id3 flatIndexToIndex3(dax::Id index,
                                                       const Extent3 &extent)
{
  dax::Id3 dims = extentDimensions(extent);

  dax::Id3 ijk;
  ijk[0] = index % dims[0];
  ijk[1] = (index / dims[0]) % dims[1];
  ijk[2] = (index / (dims[0] * dims[1]));

  return ijk + extent.Min;
}

/// Same as flatIndexToIndex3 except performed for cells using extents for
/// for points, which have one more in every direction than cells.
DAX_EXEC_CONT_EXPORT
dax::Id3 flatIndexToIndex3Cell(dax::Id index, const Extent3 &pointExtent)
{
  dax::Id3 dims = extentDimensions(pointExtent) - dax::make_Id3(1, 1, 1);

  dax::Id3 ijk;
  ijk[0] = index % dims[0];
  ijk[1] = (index / dims[0]) % dims[1];
  ijk[2] = (index / (dims[0] * dims[1]));

  return ijk + pointExtent.Min;
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts r,s,t 3d indices to a flat
/// index.
DAX_EXEC_CONT_EXPORT dax::Id index3ToFlatIndex(dax::Id3 ijk,
                                               const Extent3 &extent)
{
  dax::Id3 dims = extentDimensions(extent);
  dax::Id3 deltas = ijk - extent.Min;

  return deltas[0] + dims[0]*(deltas[1] + dims[1]*deltas[2]);
}

/// Same as index3ToFlatIndex except performed for cells using extents for
/// for points, which have one more in every direction than cells.
DAX_EXEC_CONT_EXPORT
dax::Id index3ToFlatIndexCell(dax::Id3 ijk, const Extent3 &pointExtent)
{
  dax::Id3 dims = extentDimensions(pointExtent) - dax::make_Id3(1, 1, 1);
  dax::Id3 deltas = ijk - pointExtent.Min;

  return deltas[0] + dims[0]*(deltas[1] + dims[1]*deltas[2]);
}

}

#endif //__dax__Extent_h
