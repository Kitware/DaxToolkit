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
DAX_STRUCT_ALIGN_BEGIN(DAX_SIZE_ID) struct Extent3
{
  Id3 Min;
  Id3 Max;

  DAX_EXEC_CONT_EXPORT Extent3() : Min(0), Max(0) {}

  DAX_EXEC_CONT_EXPORT Extent3( const dax::Id3& min, const dax::Id3& max):
    Min(min),
    Max(max)
    {}

  DAX_EXEC_CONT_EXPORT Extent3( const Extent3& other) :
    Min(other.Min),
    Max(other.Max)
    {}

   DAX_EXEC_CONT_EXPORT Extent3& operator= (const Extent3& other)
   {
   this->Min = other.Min;
   this->Max = other.Max;
   return *this;
   }

} __attribute__ ((aligned(DAX_SIZE_SCALAR)));

/// Given an extent, returns the array dimensions in each direction.
DAX_EXEC_CONT_EXPORT dax::Id3 extentDimensions(const Extent3 &extent)
{
  //efficient implementation that uses no temporary id3 to create dimensions
  return dax::Id3(extent.Max[0] - extent.Min[0] + 1,
                  extent.Max[1] - extent.Min[1] + 1,
                  extent.Max[2] - extent.Min[2] + 1);
}

/// Given an extent, returns the cell dimensions in each direction. instead of point
DAX_EXEC_CONT_EXPORT dax::Id3 extentCellDimensions(const Extent3 &extent)
{
  //efficient implementation that uses no temporary id3 to create dimensions
  return dax::Id3(extent.Max[0] - extent.Min[0],
                  extent.Max[1] - extent.Min[1],
                  extent.Max[2] - extent.Min[2]);
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts a flat index to the r,s,t
/// 3d indices.
DAX_EXEC_CONT_EXPORT dax::Id3 flatIndexToIndex3(dax::Id index,
                                                const Extent3 &extent)
{
  //efficient implementation that tries to reduce the number of temporary variables
  const dax::Id3 dims = extentDimensions(extent);
  return dax::Id3(
          (index % dims[0]) + extent.Min[0],
          ((index / dims[0]) % dims[1]) + extent.Min[1],
          ((index / (dims[0] * dims[1]))) + extent.Min[2]);
}

/// Same as flatIndexToIndex3 except performed for cells using extents for
/// for points, which have one more in every direction than cells.
DAX_EXEC_CONT_EXPORT
dax::Id3 flatIndexToIndex3Cell(dax::Id index, const Extent3 &extent)
{
  //efficient implementation that tries to reduce the number of temporary variables
  const dax::Id3 dims = extentCellDimensions(extent);
  return dax::Id3(
          (index % dims[0]) + extent.Min[0],
          ((index / dims[0]) % dims[1]) + extent.Min[1],
          ((index / (dims[0] * dims[1]))) + extent.Min[2]);
}

/// Elements in structured grids have a single index with 0 being the entry at
/// the minimum extent in every direction and then increasing first in the r
/// then s then t directions.  This method converts r,s,t 3d indices to a flat
/// index.
DAX_EXEC_CONT_EXPORT dax::Id index3ToFlatIndex(dax::Id3 ijk,
                                               const Extent3 &extent)
{
  const dax::Id3 dims = extentDimensions(extent);
  const dax::Id3 deltas(ijk[0] - extent.Min[0],
                        ijk[1] - extent.Min[1],
                        ijk[2] - extent.Min[2]);
  return deltas[0] + dims[0]*(deltas[1] + dims[1]*deltas[2]);
}

/// Same as index3ToFlatIndex except performed for cells using extents for
/// for points, which have one more in every direction than cells.
DAX_EXEC_CONT_EXPORT
dax::Id index3ToFlatIndexCell(dax::Id3 ijk, const Extent3 &extent)
{
  const dax::Id3 dims = extentCellDimensions(extent);
  const dax::Id3 deltas(ijk[0] - extent.Min[0],
                        ijk[1] - extent.Min[1],
                        ijk[2] - extent.Min[2]);
  return deltas[0] + dims[0]*(deltas[1] + dims[1]*deltas[2]);
}

/// Returns the first point id for a given cell index and extent
DAX_EXEC_CONT_EXPORT
dax::Id indexToConnectivityIndex(dax::Id index, const Extent3 &extent)
{
  const dax::Id3 p_dims = extentDimensions(extent);
  const dax::Id3 c_dims = extentCellDimensions(extent);
  dax::Id3 deltas(
          index % c_dims[0],
          (index / c_dims[0]) % c_dims[1],
          (index / (c_dims[0] * c_dims[1])));
  return deltas[0] + p_dims[0]*(deltas[1] + p_dims[1]*deltas[2]);
}



}

#endif //__dax__Extent_h
