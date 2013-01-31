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
#ifndef __dax__exec__internal__TopologyUniform_h
#define __dax__exec__internal__TopologyUniform_h

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/Extent.h>

#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>
namespace dax {
namespace exec {
namespace internal {

namespace detail {
/// \brief Holds the implict values to compute the point indices for a cell
///
/// This class is a convenience wrapper to delay storing implicit cells indices
///
template<class CellTag>
class ImplicitCellVertices
{
public:
  const static int NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  DAX_EXEC_EXPORT
  ImplicitCellVertices( dax::Id3 dims, dax::Id startingPointIdx ):
    XDim(dims[0]),
    FirstPointIndex(startingPointIdx),
    SecondPointIndex(startingPointIdx + (dims[0] * dims[1]) )
  {
  }

  DAX_EXEC_EXPORT
  ImplicitCellVertices( dax::Id3 dims,
                       const dax::exec::internal::IJKIndex& index ):
    XDim(dims[0])
  {
    const dax::Id3 &ijk = index.ijk();
    this->FirstPointIndex =  ijk[0] + ijk[1] * dims[0] + ijk[2] * dims[0] * dims[1];
    this->SecondPointIndex = this->FirstPointIndex + (dims[0] * dims[1]);
  }

  // Although this copy constructor should be identical to the default copy
  // constructor, we have noticed that NVCC's default copy constructor can
  // incur a significant slowdown.
  DAX_EXEC_EXPORT
  ImplicitCellVertices(const ImplicitCellVertices &src) :
    XDim(src.XDim),
    FirstPointIndex(src.FirstPointIndex),
    SecondPointIndex(src.SecondPointIndex)
    {
    }

  dax::Id XDim, FirstPointIndex, SecondPointIndex;
};
}

/// Contains all the parameters necessary to specify the topology of a uniform
/// rectilinear grid.
///
struct TopologyUniform {
  typedef dax::CellTagVoxel CellTag;

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;

  /// Returns the number of points in a uniform rectilinear grid.
  ///
  DAX_EXEC_EXPORT
  dax::Id GetNumberOfPoints() const
  {
    dax::Id3 dims = dax::extentDimensions(this->Extent);
    return dims[0]*dims[1]*dims[2];
  }

  /// Returns the number of cells in a uniform rectilinear grid.
  ///
  DAX_EXEC_EXPORT
  dax::Id GetNumberOfCells() const
  {
    dax::Id3 dims = dax::extentDimensions(this->Extent)
                    - dax::make_Id3(1, 1, 1);
    return dims[0]*dims[1]*dims[2];
  }

  /// Returns the point position in a structured grid for a given i, j, and k
  /// value stored in /c ijk
  ///
  DAX_EXEC_EXPORT
  dax::Vector3 GetPointCoordiantes(dax::Id3 ijk) const
  {
    return dax::make_Vector3(this->Origin[0] + ijk[0] * this->Spacing[0],
                             this->Origin[1] + ijk[1] * this->Spacing[1],
                             this->Origin[2] + ijk[2] * this->Spacing[2]);
  }

  /// Returns the point position in a structured grid for a given index
  /// which is represented by /c pointIndex
  ///
  DAX_EXEC_EXPORT
  dax::Vector3 GetPointCoordiantes(dax::Id pointIndex) const
  {
    dax::Id3 ijk = flatIndexToIndex3(pointIndex, this->Extent);
    return this->GetPointCoordiantes(ijk);
  }

  DAX_EXEC_EXPORT
  detail::ImplicitCellVertices<dax::CellTagVoxel>
  ComputeImplictVertices(const dax::Id& cellIndex) const
  {
    typedef detail::ImplicitCellVertices<dax::CellTagVoxel> ReturnType;
    return ReturnType( dax::extentDimensions(this->Extent),
                       indexToConnectivityIndex(cellIndex,this->Extent));
  }

  DAX_EXEC_EXPORT
  detail::ImplicitCellVertices<dax::CellTagVoxel>
  ComputeImplictVertices(const dax::exec::internal::IJKIndex& cellIndex) const
  {
  typedef detail::ImplicitCellVertices<dax::CellTagVoxel> ReturnType;
  return ReturnType(dax::extentDimensions(this->Extent), cellIndex);
  }

  template< class IndexType >
  DAX_EXEC_EXPORT
  dax::exec::CellVertices<CellTag>
  GetCellConnections(const IndexType& cellIndex) const
  {

    typedef detail::ImplicitCellVertices<CellTag> ReturnType;
    ReturnType indices = this->ComputeImplictVertices(cellIndex);

    dax::exec::CellVertices<CellTag> values;

    values[0] = indices.FirstPointIndex;
    values[1] = indices.FirstPointIndex + 1;
    values[2] = indices.FirstPointIndex + indices.XDim + 1;
    values[3] = indices.FirstPointIndex + indices.XDim;
    values[4] = indices.SecondPointIndex;
    values[5] = indices.SecondPointIndex + 1;
    values[6] = indices.SecondPointIndex + indices.XDim + 1;
    values[7] = indices.SecondPointIndex + indices.XDim;
    return values;
  }
} __attribute__ ((aligned(4)));

}  }  } //namespace dax::exec::internal

#endif //__dax__exec__internal__TopologyUniform_h
