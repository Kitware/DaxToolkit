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
template<class CellType>
struct UnstructuredGridOfCell : UnstructuredGridTag {  };

/// This class defines the topology of an unstructured grid. An unstructured
/// grid can only contain cells of a single type.
///
template <typename CellT,
          class ArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER>
class UnstructuredGrid
{
public:
  typedef CellT CellType;
  typedef UnstructuredGridOfCell<CellType> GridTypeTag;

  typedef dax::cont::ArrayHandle<
      dax::Id, ArrayContainerControlTag, DeviceAdapterTag> CellConnectionsType;
  typedef dax::cont::ArrayHandle<
      dax::Vector3, ArrayContainerControlTag, DeviceAdapterTag>
      PointCoordinatesType;

  typedef dax::exec::internal
      ::ExecutionAdapter<ArrayContainerControlTag,DeviceAdapterTag>
      ExecutionAdapter;

  typedef dax::exec::internal::TopologyUnstructured<CellType, ExecutionAdapter>
      ExecutionTopologyStruct;

private:
  CellConnectionsType CellConnections;
  PointCoordinatesType PointCoordinates;

public:
  UnstructuredGrid() { }

  UnstructuredGrid(CellConnectionsType cellConnections,
                   PointCoordinatesType pointCoordinates)
    : CellConnections(cellConnections), PointCoordinates(pointCoordinates)
  {
    DAX_ASSERT_CONT(this->CellConnections.GetNumberOfValues() % CellType::NUM_POINTS == 0);
  }

  /// The CellConnections array defines the connectivity of the mesh. The
  /// length of this array is CellType::NUM_POINTS times more than the number
  /// of cell. Each cell is represented by this number of points defining the
  /// structure of the cell.
  ///
  CellConnectionsType GetCellConnections() const {
    return this->CellConnections;
  }
  void SetCellConnections(CellConnectionsType cellConnections) {
    this->CellConnections = cellConnections;
  }


  /// The PointCoordinates array defines the location of each point.  The
  /// length of this array defines how many points are in the mesh.
  ///
  PointCoordinatesType GetPointCoordinates() const {
    return this->PointCoordinates;
  }
  void SetPointCoordinates(PointCoordinatesType pointCoordinates) {
    this->PointCoordinates = pointCoordinates;
  }

  // Helper functions

  /// Given a point idnex, computes the coordinates.
  ///
  dax::Vector3 ComputePointCoordinates(dax::Id index) const
    {
    typedef typename PointCoordinatesType::IteratorConstControl PIterator;
    DAX_ASSERT_CONT(this->PointCoordinates.GetNumberOfValues() >= index);
    DAX_ASSERT_CONT(index >= 0);

    PIterator point = this->PointCoordinates.GetIteratorConstControlBegin();
    std::advance(point,index);
    return *point;
  }

  /// Get the number of points.
  ///
  dax::Id GetNumberOfPoints() const {
    return this->PointCoordinates.GetNumberOfValues();
  }

  /// Get the number of cells.
  ///
  dax::Id GetNumberOfCells() const {
    return this->CellConnections.GetNumberOfValues() / CellType::NUM_POINTS;
  }
};

}
}

#endif //__dax_cont_UnstructuredGrid_h
