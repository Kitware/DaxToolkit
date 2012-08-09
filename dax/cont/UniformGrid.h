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
#ifndef __dax__cont__UniformGrid_h
#define __dax__cont__UniformGrid_h

#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/IteratorFromArrayPortal.h>

#include <dax/exec/CellVoxel.h>

#include <dax/exec/internal/TopologyUniform.h>

namespace dax {
namespace cont {

/// A tag you can use to identify when a grid is a uniform grid.
///
struct UniformGridTag {  };

namespace detail {

class ArrayPortalFromUniformGridPointCoordinates
{
public:
  typedef dax::Vector3 ValueType;
  typedef dax::cont::IteratorFromArrayPortal<
      ArrayPortalFromUniformGridPointCoordinates> IteratorType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalFromUniformGridPointCoordinates() {  }

  DAX_CONT_EXPORT
  ArrayPortalFromUniformGridPointCoordinates(dax::Vector3 origin,
                                             dax::Vector3 spacing,
                                             dax::Extent3 extent)
    : Origin(origin), Spacing(spacing), Extent(extent) {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    dax::Id3 dims = dax::extentDimensions(this->Extent);
    return dims[0]*dims[1]*dims[2];
  }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const {
    dax::Id3 location = dax::flatIndexToIndex3(index, this->Extent);
    return dax::make_Vector3(
          this->Origin[0] + this->Spacing[0]*location[0],
          this->Origin[1] + this->Spacing[1]*location[1],
          this->Origin[2] + this->Spacing[2]*location[2]);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const {
    return IteratorType(*this, this->GetNumberOfValues());
  }

private:
  dax::Vector3 Origin;
  dax::Vector3 Spacing;
  dax::Extent3 Extent;
};

} // namespace detail

/// This class defines the topology of a uniform grid. A uniform grid is axis
/// aligned and has uniform spacing between grid points in every dimension. The
/// grid can be shifted and scaled in space by defining and origin and spacing.
///
template <class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class UniformGrid
{
public:
  typedef dax::exec::CellVoxel CellType;
  typedef UniformGridTag GridTypeTag;

  DAX_CONT_EXPORT
  UniformGrid()
    : Origin(dax::make_Vector3(0.0, 0.0, 0.0)),
      Spacing(dax::make_Vector3(1.0, 1.0, 1.0))
  {
    this->SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(0, 0, 0));
  }

  /// The extent defines the minimum and maximum (inclusive) indices in each
  /// dimension.
  ///
  DAX_CONT_EXPORT
  const dax::Extent3 &GetExtent() const { return this->Extent; }
  void SetExtent(const dax::Extent3 &extent) { this->Extent = extent; }
  void SetExtent(const dax::Id3 &min, const dax::Id3 &max) {
    this->Extent.Min = min;
    this->Extent.Max = max;
  }

  /// The origin is the location in space of the point at grid position
  /// (0, 0, 0).  This position may or may not actually be in the extent.
  ///
  DAX_CONT_EXPORT
  const dax::Vector3 &GetOrigin() const { return this->Origin; }
  void SetOrigin(const dax::Vector3 &coords) { this->Origin = coords; }

  /// The spacing is the distance between grid points. Each component in the
  /// vector refers to a spacing along the associated axis, which can vary.
  ///
  DAX_CONT_EXPORT
  const dax::Vector3 &GetSpacing() const { return this->Spacing; }
  void SetSpacing(const dax::Vector3 &distances) { this->Spacing = distances; }

  // Helper functions

  /// Get the number of points.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfPoints() const {
    dax::Id3 dims = dax::extentDimensions(this->GetExtent());
    return dims[0]*dims[1]*dims[2];
  }

  /// Get the number of cells.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfCells() const {
    dax::Id3 dims = dax::extentDimensions(this->GetExtent())
                    - dax::make_Id3(1, 1, 1);
    return dims[0]*dims[1]*dims[2];
  }

  /// Converts an i, j, k point location to a point index.
  ///
  DAX_CONT_EXPORT
  dax::Id ComputePointIndex(const dax::Id3 &ijk) const {
    return dax::index3ToFlatIndex(ijk, this->GetExtent());
  }

  /// Converts an i, j, k cell location to a cell index.
  ///
  DAX_CONT_EXPORT
  dax::Id ComputeCellIndex(const dax::Id3 &ijk) const {
    return dax::index3ToFlatIndexCell(ijk, this->GetExtent());
  }

  /// Converts a flat point index to an i, j, k point location.
  ///
  DAX_CONT_EXPORT
  dax::Id3 ComputePointLocation(dax::Id index) const {
    return dax::flatIndexToIndex3(index, this->GetExtent());
  }

  /// Converts a flat cell index to an i, j, k cell location.
  ///
  DAX_CONT_EXPORT
  dax::Id3 ComputeCellLocation(dax::Id index) const {
    return dax::flatIndexToIndex3Cell(index, this->GetExtent());
  }

  /// Given a point i, j, k location, computes the coordinates.
  ///
  DAX_CONT_EXPORT
  dax::Vector3 ComputePointCoordinates(dax::Id3 location) const {
    dax::Vector3 coord(
          this->GetOrigin()[0] + this->GetSpacing()[0]*location[0],
          this->GetOrigin()[1] + this->GetSpacing()[1]*location[1],
          this->GetOrigin()[2] + this->GetSpacing()[2]*location[2]);
    return coord;
  }

  /// Given a point index, computes the coordinates.
  ///
  DAX_CONT_EXPORT
  dax::Vector3 ComputePointCoordinates(dax::Id index) const {
    return this->ComputePointCoordinates(this->ComputePointLocation(index));
  }

  typedef dax::cont::ArrayHandle<
      dax::Vector3,
      dax::cont::ArrayContainerControlImplicit<
          detail::ArrayPortalFromUniformGridPointCoordinates>,
      DeviceAdapterTag> PointCoordinatesType;

  DAX_CONT_EXPORT
  PointCoordinatesType GetPointCoordinates() const {
    detail::ArrayPortalFromUniformGridPointCoordinates portal(this->Origin,
                                                              this->Spacing,
                                                              this->Extent);
    return PointCoordinatesType(portal);
  }

  typedef dax::exec::internal::TopologyUniform TopologyStructConstExecution;
  typedef dax::exec::internal::TopologyUniform TopologyStructExecution;

  /// Prepares this topology to be used as an input to an operation in the
  /// execution environment.  Returns a structure that can be used directly
  /// in the execution environment.
  ///
  DAX_CONT_EXPORT
  TopologyStructConstExecution PrepareForInput() const {
    TopologyStructConstExecution topology;
    topology.Origin = this->Origin;
    topology.Spacing = this->Spacing;
    topology.Extent = this->Extent;
    return topology;
  }

private:
  dax::Vector3 Origin;
  dax::Vector3 Spacing;
  dax::Extent3 Extent;
};

}
}

#endif //__dax__cont__UniformGrid_h
