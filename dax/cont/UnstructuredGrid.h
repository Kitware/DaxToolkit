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

#ifndef __dax_cont_UnstructuredGrid_h
#define __dax_cont_UnstructuredGrid_h

#include <dax/CellTraits.h>

#include <dax/cont/ArrayHandle.h>

#include <dax/exec/internal/TopologyUnstructured.h>

namespace dax {
namespace cont {

/// A tag you can use to identify when grid is an unstructured grid.
///
struct UnstructuredGridTag {  };

/// A subtag of UnstructuredGridTag that specifies the type of cell in the grid
/// through templating.
///
template<class CellTag>
struct UnstructuredGridOfCell : UnstructuredGridTag {  };

/// This class defines the topology of an unstructured grid. An unstructured
/// grid can only contain cells of a single type.
///
template <
    typename CellT,
    class CellConnectionsContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class PointsArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class UnstructuredGrid
{
public:
  typedef CellT CellTag;
  typedef UnstructuredGridOfCell<CellTag> GridTypeTag;

  typedef dax::cont::ArrayHandle<
      dax::Id, CellConnectionsContainerControlTag, DeviceAdapterTag>
      CellConnectionsType;
  typedef dax::cont::ArrayHandle<
      dax::Vector3, PointsArrayContainerControlTag, DeviceAdapterTag>
      PointCoordinatesType;

  DAX_CONT_EXPORT
  UnstructuredGrid() { }

  DAX_CONT_EXPORT
  UnstructuredGrid(CellConnectionsType cellConnections,
                   PointCoordinatesType pointCoordinates)
    : CellConnections(cellConnections), PointCoordinates(pointCoordinates)
  {
    DAX_ASSERT_CONT((this->CellConnections.GetNumberOfValues()
                     % dax::CellTraits<CellTag>::NUM_VERTICES) == 0);
  }

  /// The CellConnections array defines the connectivity of the mesh. The
  /// length of this array is CellType::NUM_POINTS times more than the number
  /// of cell. Each cell is represented by this number of points defining the
  /// structure of the cell.
  ///
  DAX_CONT_EXPORT
  const CellConnectionsType &GetCellConnections() const {
    return this->CellConnections;
  }
  DAX_CONT_EXPORT
  CellConnectionsType &GetCellConnections() {
    return this->CellConnections;
  }
  void SetCellConnections(CellConnectionsType cellConnections) {
    this->CellConnections = cellConnections;
  }


  /// The PointCoordinates array defines the location of each point.  The
  /// length of this array defines how many points are in the mesh.
  ///
  DAX_CONT_EXPORT
  const PointCoordinatesType &GetPointCoordinates() const {
    return this->PointCoordinates;
  }
  DAX_CONT_EXPORT
  PointCoordinatesType &GetPointCoordinates() {
    return this->PointCoordinates;
  }
  DAX_CONT_EXPORT
  void SetPointCoordinates(PointCoordinatesType pointCoordinates) {
    this->PointCoordinates = pointCoordinates;
  }

  // Helper functions

  /// Given a point idnex, computes the coordinates.
  ///
  dax::Vector3 ComputePointCoordinates(dax::Id index) const{
    DAX_ASSERT_CONT(this->PointCoordinates.GetNumberOfValues() >= index);
    DAX_ASSERT_CONT(index >= 0);
    return this->PointCoordinates.GetPortalConstControl().Get(index);
  }

  /// Get the number of points.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfPoints() const {
    return this->PointCoordinates.GetNumberOfValues();
  }

  /// Get the number of cells.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfCells() const {
    return (this->CellConnections.GetNumberOfValues()
            / dax::CellTraits<CellTag>::NUM_VERTICES);
  }

  typedef dax::exec::internal::TopologyUnstructured<
      CellTag, typename CellConnectionsType::PortalConstExecution>
      TopologyStructConstExecution;

  /// Prepares this topology to be used as an input to an operation in the
  /// execution environment. Returns a structure that can be used directly in
  /// the execution environment.
  ///
  DAX_CONT_EXPORT
  TopologyStructConstExecution PrepareForInput() const {
    return TopologyStructConstExecution(this->CellConnections.PrepareForInput(),
                                        this->GetNumberOfPoints(),
                                        this->GetNumberOfCells());
  }

  typedef dax::exec::internal::TopologyUnstructured<
      CellTag, typename CellConnectionsType::PortalExecution>
      TopologyStructExecution;

  /// Prepares this topology to be used as an output to an operation that
  /// generates topology connections. Returns a structure that has a
  /// connections array that can be set. Point coordinates are not considered
  /// here and have to be set separately.
  ///
  DAX_CONT_EXPORT
  TopologyStructExecution PrepareForOutput(dax::Id numberOfCells) {
    // Set the number of points to 0 since we really don't know better. Now
    // that I consider it, I wonder what the point of having the number of
    // points field in the first place. The number of cells fields seems pretty
    // redundant, too.
    return TopologyStructExecution(
          this->CellConnections.PrepareForOutput(
            numberOfCells*dax::CellTraits<CellTag>::NUM_VERTICES),
          0,
          numberOfCells);
  }

private:
  CellConnectionsType CellConnections;
  PointCoordinatesType PointCoordinates;
};

}
}

#endif //__dax_cont_UnstructuredGrid_h
