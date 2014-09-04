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

#ifndef __dax_cont_internal_EdgeInterpolatedGrid_h
#define __dax_cont_internal_EdgeInterpolatedGrid_h

#include <dax/CellTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/GridTags.h>
#include <dax/exec/internal/TopologyUnstructured.h>

namespace dax {

struct PointAsEdgeInterpolation
{
dax::Id EdgeIdFirst;
dax::Id EdgeIdSecond;
dax::Scalar Weight;


DAX_EXEC_CONT_EXPORT
PointAsEdgeInterpolation():
  EdgeIdFirst(0),
  EdgeIdSecond(0),
  Weight(0.0f)
  {
  }

DAX_EXEC_CONT_EXPORT
PointAsEdgeInterpolation(dax::Id first, dax::Id second, dax::Scalar w):
  EdgeIdFirst(first),
  EdgeIdSecond(second),
  Weight(w)
  {
  }


DAX_EXEC_CONT_EXPORT
bool operator ==( const PointAsEdgeInterpolation& other) const
{
return (this->EdgeIdFirst == other.EdgeIdFirst &&
        this->EdgeIdSecond == other.EdgeIdSecond);
}

DAX_EXEC_CONT_EXPORT
bool operator !=( const PointAsEdgeInterpolation& other) const
{
return (this->EdgeIdFirst != other.EdgeIdFirst ||
        this->EdgeIdSecond != other.EdgeIdSecond);
}

DAX_EXEC_CONT_EXPORT
bool operator<(const dax::PointAsEdgeInterpolation& other) const
{
return (this->EdgeIdFirst < other.EdgeIdFirst) ||
       (this->EdgeIdFirst == other.EdgeIdFirst &&
          this->EdgeIdSecond < other.EdgeIdSecond);
}

};



namespace cont {
namespace internal {

/// This class defines the topology of an unstructured grid. An unstructured
/// grid can only contain cells of a single type.
///
template <
    typename CellT,
    class CellConnectionsContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class PointsArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class EdgeInterpolatedGrid
{
public:
  typedef CellT CellTag;
  typedef dax::cont::internal::UnspecifiedGridTag GridTypeTag;

  typedef dax::cont::ArrayHandle< dax::Id,
                  CellConnectionsContainerControlTag, DeviceAdapterTag>
      CellConnectionsType;
  typedef dax::cont::ArrayHandle< dax::PointAsEdgeInterpolation,
                  PointsArrayContainerControlTag, DeviceAdapterTag>
      InterpolatedPointsType;

  DAX_CONT_EXPORT
  EdgeInterpolatedGrid() { }

  DAX_CONT_EXPORT
  EdgeInterpolatedGrid(CellConnectionsType cellConnections,
                       InterpolatedPointsType pointCoordinates)
    : CellConnections(cellConnections), InterpolatedPoints(pointCoordinates)
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
  const InterpolatedPointsType &GetInterpolatedPoints() const {
    return this->InterpolatedPoints;
  }
  DAX_CONT_EXPORT
  InterpolatedPointsType &GetInterpolatedPoints() {
    return this->InterpolatedPoints;
  }
  DAX_CONT_EXPORT
  void SetInterpolatedPoints(InterpolatedPointsType pointCoordinates) {
    this->InterpolatedPoints = pointCoordinates;
  }

  // Helper functions

  /// Get the number of points.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfPoints() const {
    return this->InterpolatedPoints.GetNumberOfValues();
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
  TopologyStructExecution PrepareForOutput(dax::Id numberOfInterpolatedEdges) {
    // Set the number of points to 0 since we really don't know better. Now
    // that I consider it, I wonder what the point of having the number of
    // points field in the first place. The number of cells fields seems pretty
    // redundant, too.
    return TopologyStructExecution(
          this->CellConnections.PrepareForOutput(
            numberOfInterpolatedEdges*dax::CellTraits<CellTag>::NUM_VERTICES),
          0,
          numberOfInterpolatedEdges);
  }

private:
  CellConnectionsType CellConnections;
  InterpolatedPointsType InterpolatedPoints;
};

}
}
}

#endif //__dax_cont_internal_EdgeInterpolatedGrid_h
