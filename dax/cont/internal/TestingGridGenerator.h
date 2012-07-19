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

#ifndef __dax_cont_internal_TestingGridGenerator_h
#define  __dax_cont_internal_TestingGridGenerator_h

#include <dax/Types.h>

#include <dax/exec/Cell.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <vector>

#include <iostream>

namespace dax {
namespace cont {
namespace internal {

template<class GridType,
         class ArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL,
         class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER>
struct TestGrid
{
private:
  //custom storage container that changes the size based on the grid type
  //so that we don't have to store extra information for uniform grid
  const dax::Id Size;
  GridType Grid;
  template<typename GT> struct GridStorage {};
  template<typename U> struct GridStorage<dax::cont::UnstructuredGrid<U> >
    {
    std::vector<dax::Id> topology;
    std::vector<dax::Vector3> points;
    };
  GridStorage<GridType> Info;

public:

  TestGrid(const dax::Id& size)
    :Size(size),
     Grid(),
     Info()
    {
    this->BuildGrid(this->Grid);
    }

  /// Enable pointer-like dereference syntax. Returns a pointer to the
  /// contained object.
  ///
  const GridType* operator->() const
    {
    return &this->Grid;
    }

  /// Get a raw pointer to the contained object.
  ///
  const GridType& GetRealGrid() const
    {
    return this->Grid;
    }

  /// This convienience function allows you to generate the point coordinate
  /// field (since there is no consistent way to get it from the grid itself.
  ///
  dax::Vector3 GetPointCoordinates(dax::Id index)
  {
    // Not an efficient implementation, but it's test code so who cares?
    dax::cont::UniformGrid<> uniform;
    this->BuildGrid(uniform);
    return uniform.ComputePointCoordinates(index);
  }

  ~TestGrid()
  {
    std::cout << "Test grid destroyed.  "
              << "Any use of the grid after this point is an error."
              << std::endl;
  }

private:
  template<typename T>
  dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
  MakeArrayHandle(const std::vector<T> &array)
  {
    return dax::cont::make_ArrayHandle(array,
                                       ArrayContainerControlTag(),
                                       DeviceAdapterTag());
  }

  void BuildGrid(dax::cont::UniformGrid<> &grid)
    {
    grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(Size-1, Size-1, Size-1));
    }

  void BuildGrid(dax::cont::UnstructuredGrid<dax::exec::CellTriangle> &grid)
    {
    //we need to make a volume grid
    dax::cont::UniformGrid<> uniform;
    uniform.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(Size-1, Size-1, Size-1));

    //copy the point info over to the unstructured grid
    this->Info.points.clear();
    for(dax::Id i=0; i <uniform.GetNumberOfPoints(); ++i)
      {
      this->Info.points.push_back(uniform.ComputePointCoordinates(i));
      }

    //copy the cell topology information over
    //create 2 triangles for each of the 6 sides of cube
    const dax::Id3 cellVertexToPointIndex[36] = {
      // Front
      dax::make_Id3(0, 0, 0),
      dax::make_Id3(1, 0, 0),
      dax::make_Id3(1, 1, 0),

      dax::make_Id3(1, 1, 0),
      dax::make_Id3(0, 1, 0),
      dax::make_Id3(0, 0, 0),

      // Left
      dax::make_Id3(0, 0, 0),
      dax::make_Id3(0, 1, 0),
      dax::make_Id3(0, 1, 1),

      dax::make_Id3(0, 1, 1),
      dax::make_Id3(0, 0, 1),
      dax::make_Id3(0, 0, 0),

      // Bottom
      dax::make_Id3(0, 0, 0),
      dax::make_Id3(0, 0, 1),
      dax::make_Id3(1, 0, 1),

      dax::make_Id3(1, 0, 1),
      dax::make_Id3(1, 0, 0),
      dax::make_Id3(0, 0, 0),

      // Back
      dax::make_Id3(1, 1, 1),
      dax::make_Id3(1, 0, 1),
      dax::make_Id3(0, 0, 1),

      dax::make_Id3(0, 0, 1),
      dax::make_Id3(0, 1, 1),
      dax::make_Id3(1, 1, 1),

      // Right
      dax::make_Id3(1, 1, 1),
      dax::make_Id3(1, 1, 0),
      dax::make_Id3(0, 1, 0),

      dax::make_Id3(0, 1, 0),
      dax::make_Id3(0, 1, 1),
      dax::make_Id3(1, 1, 1),

      // Top
      dax::make_Id3(1, 1, 1),
      dax::make_Id3(1, 0, 1),
      dax::make_Id3(1, 0, 0),

      dax::make_Id3(1, 0, 0),
      dax::make_Id3(1, 1, 0),
      dax::make_Id3(1, 1, 1),
    };

    this->Info.topology.clear();
    dax::Id numPointsPerCell = ::dax::exec::CellTriangle::NUM_POINTS;
    const dax::Extent3 extents = uniform.GetExtent();
    for ( dax::Id i=0; i < uniform.GetNumberOfCells(); ++i )
      {
      dax::Id3 ijkCell = dax::flatIndexToIndex3Cell( i, extents );
      for(dax::Id j=0; j < numPointsPerCell*12; ++j)
        {
        dax::Id3 ijkPoint = ijkCell + cellVertexToPointIndex[j];

        dax::Id pointIndex = index3ToFlatIndex(ijkPoint,extents);
        this->Info.topology.push_back(pointIndex);
        }
      }

    grid = dax::cont::UnstructuredGrid<dax::exec::CellTriangle>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  void BuildGrid(dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> &grid)
    {
    //we need to make a volume grid
    dax::cont::UniformGrid<> uniform;
    this->BuildGrid(uniform);

    //copy the point info over to the unstructured grid
    this->Info.points.clear();
    for(dax::Id i=0; i <uniform.GetNumberOfPoints(); ++i)
      {
      this->Info.points.push_back(uniform.ComputePointCoordinates(i));
      }

    //copy the cell topology information over

    //this only works for voxel/hexahedron
    const dax::Id3 cellVertexToPointIndex[8] = {
      dax::make_Id3(0, 0, 0),
      dax::make_Id3(1, 0, 0),
      dax::make_Id3(1, 1, 0),
      dax::make_Id3(0, 1, 0),
      dax::make_Id3(0, 0, 1),
      dax::make_Id3(1, 0, 1),
      dax::make_Id3(1, 1, 1),
      dax::make_Id3(0, 1, 1)
    };

    this->Info.topology.clear();
    dax::Id numPointsPerCell = ::dax::exec::CellHexahedron::NUM_POINTS;
    const dax::Extent3 extents = uniform.GetExtent();
    for(dax::Id i=0; i <uniform.GetNumberOfCells(); ++i)
      {
      dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(i,extents);
      for(dax::Id j=0; j < numPointsPerCell; ++j)
        {
        dax::Id3 ijkPoint = ijkCell + cellVertexToPointIndex[j];

        dax::Id pointIndex = index3ToFlatIndex(ijkPoint,extents);
        this->Info.topology.push_back(pointIndex);
        }
      }

    grid = dax::cont::UnstructuredGrid<dax::exec::CellHexahedron>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }
};

}
}
} //namespace dax::cont::internal

#endif //  __dax_cont_internal_TestingGridGenerator_h
