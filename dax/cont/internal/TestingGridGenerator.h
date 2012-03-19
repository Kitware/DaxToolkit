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

namespace dax {
namespace cont {
namespace internal {

template<typename T, typename DA>
struct TestGrid
{
  typedef T GridType;
  typedef DA DeviceAdapter;

private:
  //custom storage container that changes the size based on the grid type
  //so that we don't have to store extra information for uniform grid
  const dax::Id Size;
  GridType Grid;
  template<typename GridType> struct GridStorage {};
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

  // Description:
  // Enable pointer-like dereference syntax. Returns a pointer to the contained
  // object.
  const GridType* operator->() const
    {
    return &this->Grid;
    }

  // Description:
  // Get a raw pointer to the contained object.
  const GridType& GetRealGrid() const
    {
    return this->Grid;
    }

private:
  void BuildGrid(dax::cont::UniformGrid &grid)
    {
    grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(Size-1, Size-1, Size-1));
    }

  void BuildGrid(dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> &grid)
    {
    //we need to make a volume grid
    dax::cont::UniformGrid uniform;
    uniform.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(Size-1, Size-1, Size-1));

    //copy the point info over to the unstructured grid
    this->Info.points.clear();
    for(dax::Id i=0; i <uniform.GetNumberOfPoints(); ++i)
      {
      this->Info.points.push_back(uniform.GetPointCoordinates(i));
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

    dax::cont::ArrayHandle<dax::Vector3,DeviceAdapter> ahPoints(
          this->Info.points.begin(),this->Info.points.end());
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> ahTopo(
          this->Info.topology.begin(),this->Info.topology.end());
    grid = dax::cont::UnstructuredGrid<dax::exec::CellHexahedron>(ahTopo,
                                                                  ahPoints);
    }
};

}
}
} //namespace dax::cont::internal

#endif //  __dax_cont_internal_TestingGridGenerator_h
