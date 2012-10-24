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
#define __dax_cont_internal_TestingGridGenerator_h

#include <dax/Types.h>

#include <dax/exec/Cell.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <vector>

#include <iostream>

namespace dax {
namespace cont {
namespace internal {

template<
    class GridType,
    class ArrayContainerControlTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG,
    class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
struct TestGrid
{
private:
  //custom storage container that changes the size based on the grid type
  //so that we don't have to store extra information for uniform grid
  const dax::Id Size;
  GridType Grid;
  template<typename GT> struct GridStorage {};
  template<typename U, class UCCT, class DAT>
  struct GridStorage<dax::cont::UnstructuredGrid<U,UCCT,UCCT,DAT> >
    {
    std::vector<dax::Id> topology;
    std::vector<dax::Vector3> points;
    };
  GridStorage<GridType> Info;

public:
  typedef typename GridType::CellType CellType;
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
    return this->Grid.ComputePointCoordinates(index);
  }


  //Get the cell at a given index
  CellType GetCell(dax::Id index) const
  {
    typedef typename GridType::TopologyStructConstExecution TopoType;
    TopoType topo = this->Grid.PrepareForInput();
    return CellType(topo,index);
  }

  //get the cell connections (aka topology) at a given cell id
  dax::Tuple<dax::Id, CellType::NUM_POINTS> GetCellConnections(dax::Id cellId) const
  {
    CellType c = this->GetCell(cellId);
    return c.GetPointIndices();
  }

  /// This convienience function allows you to generate the Cell
  // point coordinates for any given data set
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> GetCellVertexCoordinates(dax::Id cellIndex) const
  {
    typedef typename GridType::PointCoordinatesType CoordType;

    //get the point ids for this cell
    dax::Tuple<dax::Id, CellType::NUM_POINTS> cellConnections =
        this->GetCellConnections(cellIndex);

    //get all the points for data set
    CoordType allCoords = this->Grid.GetPointCoordinates();

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> coordinates;
    for (dax::Id index = 0; index < CellType::NUM_POINTS; index++)
      {
      coordinates[index] = allCoords.GetPortalConstControl().Get(cellConnections[index]);
      }
    return coordinates;
  }

  ~TestGrid()
  {
    std::cout << "Test grid destroyed.  "
              << "Any use of the grid after this point is an error."
              << std::endl;
  }

private:

  // .......................................................... MakeArrayHandle
  template<typename T>
  dax::cont::ArrayHandle<T, ArrayContainerControlTag, DeviceAdapterTag>
  MakeArrayHandle(const std::vector<T> &array)
  {
    return dax::cont::make_ArrayHandle(array,
                                       ArrayContainerControlTag(),
                                       DeviceAdapterTag());
  }

  // ........................................................... MakeInfoPoints
  void MakeInfoPoints(dax::cont::UniformGrid<DeviceAdapterTag> &uniform)
    {
      //copy the point info over to the unstructured grid
      this->Info.points.clear();
      for(dax::Id i=0; i <uniform.GetNumberOfPoints(); ++i)
        {
        this->Info.points.push_back(uniform.ComputePointCoordinates(i));
        }
    }

  // ......................................................... MakeInfoTopology
  void MakeInfoTopology(dax::cont::UniformGrid<DeviceAdapterTag> &uniform,
                        const dax::Id vertexIdList[],
                        dax::Id numPointsPerCell,
                        dax::Id totalCells)
    {
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
      const dax::Extent3 extents = uniform.GetExtent();
      for(dax::Id i=0; i <uniform.GetNumberOfCells(); ++i)
        {
        dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(i,extents);
        for(dax::Id j=0; j < numPointsPerCell*totalCells; ++j)
          {
          dax::Id3 ijkPoint = ijkCell + cellVertexToPointIndex[vertexIdList[j]];

          dax::Id pointIndex = index3ToFlatIndex(ijkPoint,extents);
          this->Info.topology.push_back(pointIndex);
          }
        }
    }

  // .............................................................. UniformGrid
  void BuildGrid(dax::cont::UniformGrid<DeviceAdapterTag> &grid)
    {
    grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(Size-1, Size-1, Size-1));
    }

  // ............................................................... Hexahedron
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellHexahedron,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      //we need to make a volume grid
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      //copy the point info over to the unstructured grid
      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellHexahedron::NUM_POINTS;
      dax::Id totalCells = 1;
      const dax::Id vertexIdList[] =
        {
          0,1,2,3,4,5,6,7
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellHexahedron,
             ArrayContainerControlTag,
             ArrayContainerControlTag,
             DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // .............................................................. Tetrahedron
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellTetrahedron,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellTetrahedron::NUM_POINTS;
      dax::Id totalCells = 2;
      const dax::Id vertexIdList[] =
        {
          0,1,4,3,                // Front-Low-Left
          6,7,2,5                 // Back-Top-Right
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellTetrahedron,
             ArrayContainerControlTag,
             ArrayContainerControlTag,
             DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // .................................................................... Wedge
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellWedge,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellWedge::NUM_POINTS;
      dax::Id totalCells = 2;
      const dax::Id vertexIdList[] =
        {
          4,5,6,0,1,2,                // right-quad -> left bottom edge
          2,3,0,6,7,4                 // left-quad  -> right top edge
        };
      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellWedge,
             ArrayContainerControlTag,
             ArrayContainerControlTag,
             DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // ................................................................. Triangle
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellTriangle,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      //we need to make a volume grid
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellTriangle::NUM_POINTS;
      dax::Id totalCells = 12;
      const dax::Id vertexIdList[] =
        {
          0,1,2,2,3,0,                    // Front
          0,3,7,7,4,0,                    // Left
          0,4,5,5,1,0,                    // Bottom
          6,5,4,4,7,6,                    // Back
          6,2,3,3,7,6,                    // Top
          6,5,1,1,2,6                     // Right
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellTriangle,
             ArrayContainerControlTag,
             ArrayContainerControlTag,
             DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // ............................................................ Quadrilateral
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellQuadrilateral,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellQuadrilateral::NUM_POINTS;
      dax::Id totalCells = 6;
      const dax::Id vertexIdList[] =
        {
          0,1,2,3,              // Front
          0,3,7,4,              // Left
          0,4,5,1,              // Bottom
          6,5,4,7,              // Back
          6,2,3,7,              // Top
          6,5,1,2               // Right
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellQuadrilateral,
             ArrayContainerControlTag,
             ArrayContainerControlTag,
             DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // ..................................................................... Line
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellLine,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellLine::NUM_POINTS;
      dax::Id totalCells = 12;
      const dax::Id vertexIdList[] =
        {
          0,1,1,2,2,3,3,0,      // Front
          0,3,3,7,7,4,4,0,      // Left
          0,4,4,5,5,1,1,0,      // Bottom
          6,5,5,4,4,7,7,6,      // Back
          6,2,2,3,3,7,7,6,      // Top
          6,5,5,1,1,2,2,6       // Right
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
             dax::exec::CellLine,
              ArrayContainerControlTag,
              ArrayContainerControlTag,
              DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }

  // ................................................................... Vertex
  void BuildGrid(
    dax::cont::UnstructuredGrid<
    dax::exec::CellVertex,
    ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
    &grid)
    {
      dax::cont::UniformGrid<DeviceAdapterTag> uniform;
      this->BuildGrid(uniform);

      this->MakeInfoPoints(uniform);

      dax::Id numPointsPerCell = ::dax::exec::CellVertex::NUM_POINTS;
      dax::Id totalCells = 8;
      const dax::Id vertexIdList[] =
        {
          0,1,2,3,4,5,6,7
        };

      this->MakeInfoTopology(uniform,
                             vertexIdList,
                             numPointsPerCell,
                             totalCells);

      grid = dax::cont::UnstructuredGrid<
              dax::exec::CellVertex,
              ArrayContainerControlTag,
              ArrayContainerControlTag,
              DeviceAdapterTag>(
          this->MakeArrayHandle(this->Info.topology),
          this->MakeArrayHandle(this->Info.points));
    }
};


struct GridTesting
{
  /// Check functors to be used with the TryAllTypes method.
  ///
  struct TypeCheckAlwaysTrue {
    template <typename T, class Functor>
    void operator()(T t, Functor function) const { function(t); }
  };

  struct TypeCheckUniformGrid {
    template <typename T, class Functor>
    void operator()(T daxNotUsed(t), Functor daxNotUsed(function)) const {  }

    template<class DeviceAdapterTag, class Functor>
    void operator()(dax::cont::UniformGrid<DeviceAdapterTag> t,
                    Functor function) const { function(t); }
  };

  template<class FunctionType>
  struct InternalPrintOnInvoke {
    InternalPrintOnInvoke(FunctionType function, std::string toprint)
      : Function(function), ToPrint(toprint) { }
    template <typename T> void operator()(T t) {
      std::cout << this->ToPrint << std::endl;
      this->Function(t);
    }
  private:
    FunctionType Function;
    std::string ToPrint;
  };

  /// Runs templated \p function on all the grid types defined in Dax. This is
  /// helpful to test templated functions that should work on all grid types. If the
  /// function is supposed to work on some subset of grids or cells, then \p check can
  /// be set to restrict the types used. This Testing class contains several
  /// helpful check functors.
  ///
  template<class FunctionType,
           class CheckType,
           class ArrayContainerControlTag,
           class DeviceAdapterTag>
  static void TryAllGridTypes(FunctionType function,
                              CheckType check,
                              ArrayContainerControlTag,
                              DeviceAdapterTag)
  {
    dax::cont::UniformGrid<DeviceAdapterTag> grid;
    check(grid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UniformGrid"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellHexahedron,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        hexGrid;
    check(hexGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Hexahedron"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellTetrahedron,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        tetGrid;
    check(tetGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Tetrahedrons"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellWedge,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        wedgeGrid;
    check(wedgeGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Wedges"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellTriangle,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        triGrid;
    check(triGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Triangles"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellQuadrilateral,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        quadGrid;
    check(quadGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Quadrilaterals"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellLine,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        lineGrid;
    check(lineGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Lines"));

    dax::cont::UnstructuredGrid<
        dax::exec::CellVertex,
        ArrayContainerControlTag,ArrayContainerControlTag,DeviceAdapterTag>
        vertGrid;
    check(vertGrid, InternalPrintOnInvoke<FunctionType>(
            function, "dax::UnstructuredGrid of Vertices"));

  }
  template<class FunctionType,
           class ArrayContainerControlTag,
           class DeviceAdapterTag>
  static void TryAllGridTypes(FunctionType function,
                              ArrayContainerControlTag,
                              DeviceAdapterTag)
  {
    TryAllGridTypes(function,
                    TypeCheckAlwaysTrue(),
                    ArrayContainerControlTag(),
                    DeviceAdapterTag());
  }
  template<class FunctionType, class CheckType>
  static void TryAllGridTypes(FunctionType function, CheckType check)
  {
    TryAllGridTypes(function,
                    check,
                    DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG(),
                    DAX_DEFAULT_DEVICE_ADAPTER_TAG());
  }
  template<class FunctionType>
  static void TryAllGridTypes(FunctionType function)
  {
    TryAllGridTypes(function, TypeCheckAlwaysTrue());
  }

};


}
}
} //namespace dax::cont::internal

#endif //  __dax_cont_internal_TestingGridGenerator_h
