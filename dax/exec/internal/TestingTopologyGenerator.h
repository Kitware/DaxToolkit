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
#ifndef __dax_exec_internal_TestingTopologyGenerator_h
#define __dax_exec_internal_TestingTopologyGenerator_h

#include <dax/Types.h>

#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>

#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>

// This use of the STL vector only works in non-CUDA unit tests.
#include <vector>

namespace dax {
namespace exec {
namespace internal {

template<class TopologyType>
class TestTopology
{
public:
  typedef typename TopologyType::CellType CellType;
  typedef TestExecutionAdapter ExecutionAdapter;

private:
  TopologyType Topology;
  std::vector<dax::Vector3> CoordinatesArray;
  std::vector<dax::Id> ConnectionsArray;

public:
  TestTopology() {
    this->BuildTopology(this->Topology,
                        this->CoordinatesArray,
                        this->ConnectionsArray);
  }

  dax::Id GetNumberOfPoints() const {
    return numberOfPoints(this->Topology);
  }
  dax::Id GetNumberOfCells() const {
    return numberOfCells(this->Topology);
  }

  TopologyType GetTopology() const { return this->Topology; }

  FieldCoordinatesIn<ExecutionAdapter> GetCoordinates() const {
    return
        FieldCoordinatesIn<ExecutionAdapter>(&this->CoordinatesArray.front());
  }

  dax::Tuple<dax::Id, CellType::NUM_POINTS>
  GetCellConnections(dax::Id cellId) const
  {
    return TestTopology::GetCellConnectionsImpl(this->Topology, cellId);
  }

private:
  static dax::exec::internal::TopologyUniform GetCoreTopology()
  {
    dax::exec::internal::TopologyUniform topology;
    topology.Origin = dax::make_Vector3(1.0, -0.5, 13.0);
    topology.Spacing = dax::make_Vector3(2.5, 6.25, 1.0);
    topology.Extent.Min = dax::make_Id3(5, -2, -7);
    topology.Extent.Max = dax::make_Id3(20, 4, 10);
    return topology;
  }

  static dax::Tuple<dax::Id,8> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUniform &topology,
      dax::Id flatCellIndex)
  {
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
    const dax::Extent3 extent = topology.Extent;

    dax::Tuple<dax::Id,8> connections;

    dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(flatCellIndex, extent);
    for(dax::Id relativePointIndex = 0;
        relativePointIndex < 8;
        relativePointIndex++)
      {
      dax::Id3 ijkPoint = ijkCell+cellVertexToPointIndex[relativePointIndex];

      dax::Id pointIndex = index3ToFlatIndex(ijkPoint, extent);
      connections[relativePointIndex] = pointIndex;
      }
    return connections;
  }

  static void BuildTopology(dax::exec::internal::TopologyUniform &topology,
                            std::vector<dax::Vector3> &/*coords*/,
                            std::vector<dax::Id> &/*connections*/)
  {
    topology = TestTopology::GetCoreTopology();
  }

  static dax::Tuple<dax::Id,8> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUnstructured<
          dax::exec::CellHexahedron,TestExecutionAdapter> &,
      dax::Id cellIndex)
  {
    return TestTopology::GetCellConnectionsImpl(TestTopology::GetCoreTopology(),
                                                cellIndex);
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellHexahedron, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the uniform topology.
    dax::exec::internal::TopologyUniform uniformTopology =
        TestTopology::GetCoreTopology();

    typedef dax::exec::CellHexahedron CellType;

    topology.NumberOfCells =
        dax::exec::internal::numberOfCells(uniformTopology);
    topology.NumberOfPoints =
        dax::exec::internal::numberOfPoints(uniformTopology);

    coordArray.resize(topology.NumberOfPoints);

    // Make point coordiantes.
    for (dax::Id pointIndex = 0;
         pointIndex < topology.NumberOfPoints;
         pointIndex++)
      {
      dax::Vector3 coord =
          dax::exec::internal::pointCoordiantes(uniformTopology, pointIndex);
      // Perterb coordinates so that they are not axis aligned.
      coord[0] += dax::Scalar(0.5)*coord[2];
      coord[1] += coord[2];
      coordArray[pointIndex] = coord;
      }

    // Make connections
    connectArray.reserve(topology.NumberOfCells*CellType::NUM_POINTS);

    for(dax::Id flatCellIndex = 0;
        flatCellIndex < topology.NumberOfCells;
        flatCellIndex++)
      {
      dax::Tuple<dax::Id,CellType::NUM_POINTS> pointConnections =
          TestTopology::GetCellConnectionsImpl(uniformTopology, flatCellIndex);
      for(dax::Id relativePointIndex = 0;
          relativePointIndex < CellType::NUM_POINTS;
          relativePointIndex++)
        {
        connectArray.push_back(pointConnections[relativePointIndex]);
        }
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
  }

  static dax::Tuple<dax::Id,4> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUnstructured<
          dax::exec::CellTetrahedron,TestExecutionAdapter> &,
      dax::Id cellIndex)
  {
    // Tetrahedron meshes are a Freudenthal subdivision of hexahedra.
    dax::Tuple<dax::Id,8> hexConnections =
        TestTopology::GetCellConnectionsImpl(TestTopology::GetCoreTopology(),
                                             cellIndex/6);
    dax::Tuple<dax::Id,4> tetConnections;
    switch (cellIndex%6)
      {
      case 0:
        tetConnections[0] = hexConnections[0];
        tetConnections[1] = hexConnections[1];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      case 1:
        tetConnections[0] = hexConnections[1];
        tetConnections[1] = hexConnections[5];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      case 2:
        tetConnections[0] = hexConnections[5];
        tetConnections[1] = hexConnections[6];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      case 3:
        tetConnections[0] = hexConnections[6];
        tetConnections[1] = hexConnections[7];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      case 4:
        tetConnections[0] = hexConnections[7];
        tetConnections[1] = hexConnections[3];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      case 5:
        tetConnections[0] = hexConnections[3];
        tetConnections[1] = hexConnections[0];
        tetConnections[2] = hexConnections[2];
        tetConnections[3] = hexConnections[4];
        break;
      }
    return tetConnections;
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellTetrahedron, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the hexahedron topology.
    dax::exec::internal::TopologyUnstructured<
        dax::exec::CellHexahedron,TestExecutionAdapter> hexTopology;
    std::vector<dax::Id> hexConnections;
    BuildTopology(hexTopology, coordArray, hexConnections);

    typedef dax::exec::CellTetrahedron CellType;

    // Our tetrahedron mesh will be a Freudenthal tetrahedronization of a
    // hexahedron mesh.  This results in 6 tetrahedra for each hexahedron.
    // It is not minimal, but good enough for tests.
    topology.NumberOfCells = hexTopology.NumberOfCells * 6;
    topology.NumberOfPoints = hexTopology.NumberOfPoints;

    // Make connections.
    connectArray.reserve(topology.NumberOfCells*CellType::NUM_POINTS);

    for (dax::Id cellIndex = 0; cellIndex < topology.NumberOfCells; cellIndex++)
      {
      dax::Tuple<dax::Id,CellType::NUM_POINTS> pointConnections
          = TestTopology::GetCellConnectionsImpl(topology, cellIndex);

      connectArray.push_back(pointConnections[0]);
      connectArray.push_back(pointConnections[1]);
      connectArray.push_back(pointConnections[2]);
      connectArray.push_back(pointConnections[3]);
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
  }

  static dax::Tuple<dax::Id,6> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUnstructured<
          dax::exec::CellWedge,TestExecutionAdapter> &,
      dax::Id cellIndex)
  {
    // Wedge meshes are based on hex meshes.  They have 2x the cells.
    // See the comment in BuildTopology.
    dax::Tuple<dax::Id,8> hexConnections =
        TestTopology::GetCellConnectionsImpl(TestTopology::GetCoreTopology(),
                                             cellIndex/2);
    dax::Tuple<dax::Id,6> wedgeConnections;
    if (cellIndex%2 == 0)
      {
      wedgeConnections[0] = hexConnections[0];
      wedgeConnections[1] = hexConnections[3];
      wedgeConnections[2] = hexConnections[2];
      wedgeConnections[3] = hexConnections[4];
      wedgeConnections[4] = hexConnections[7];
      wedgeConnections[5] = hexConnections[6];
      }
    else // cellIndex%2 == 1
      {
      wedgeConnections[0] = hexConnections[0];
      wedgeConnections[1] = hexConnections[2];
      wedgeConnections[2] = hexConnections[1];
      wedgeConnections[3] = hexConnections[4];
      wedgeConnections[4] = hexConnections[6];
      wedgeConnections[5] = hexConnections[5];
      }
    return wedgeConnections;
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellWedge, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the hexahedron topology.
    dax::exec::internal::TopologyUnstructured<
        dax::exec::CellHexahedron,TestExecutionAdapter> hexTopology;
    std::vector<dax::Id> hexConnections;
    BuildTopology(hexTopology, coordArray, hexConnections);

    typedef dax::exec::CellWedge CellType;

    // Our wedge mesh will have two wedges for each hexahedron. The wedges are
    // formed by cutting the hexahedron along the x/y diagonal.
    topology.NumberOfCells = hexTopology.NumberOfCells * 2;
    topology.NumberOfPoints = hexTopology.NumberOfPoints;

    // Make connections.
    connectArray.reserve(topology.NumberOfCells*CellType::NUM_POINTS);

    for (dax::Id cellIndex = 0; cellIndex < topology.NumberOfCells; cellIndex++)
      {
      dax::Tuple<dax::Id,CellType::NUM_POINTS> pointConnections
          = TestTopology::GetCellConnectionsImpl(topology, cellIndex);

      connectArray.push_back(pointConnections[0]);
      connectArray.push_back(pointConnections[1]);
      connectArray.push_back(pointConnections[2]);
      connectArray.push_back(pointConnections[3]);
      connectArray.push_back(pointConnections[4]);
      connectArray.push_back(pointConnections[5]);
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
  }

  static dax::Tuple<dax::Id,3> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUnstructured<
          dax::exec::CellTriangle,TestExecutionAdapter> &,
      dax::Id cellIndex)
  {
    // Triangle meshes are based on hex meshes.  They have 2x the cells.
    // See the comment in BuildTopology.
    dax::Tuple<dax::Id,8> hexConnections =
        TestTopology::GetCellConnectionsImpl(TestTopology::GetCoreTopology(),
                                             cellIndex/2);
    dax::Tuple<dax::Id,3> triConnections;
    if (cellIndex%2 == 0)
      {
      triConnections[0] = hexConnections[0];
      triConnections[1] = hexConnections[5];
      triConnections[2] = hexConnections[7];
      }
    else // cellIndex%2 == 1
      {
      triConnections[0] = hexConnections[6];
      triConnections[1] = hexConnections[3];
      triConnections[2] = hexConnections[1];
      }
    return triConnections;
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellTriangle, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the hexahedron topology.
    dax::exec::internal::TopologyUnstructured<
        dax::exec::CellHexahedron,TestExecutionAdapter> hexTopology;
    std::vector<dax::Id> hexConnections;
    BuildTopology(hexTopology, coordArray, hexConnections);

    typedef dax::exec::CellTriangle CellType;

    // Our triangle mesh will have two triangles for each hexahedron. The edges
    // of the triangles come from diagonals of the faces. The two triangles do
    // not touch either and (as follows by pigeonhole principle) all faces have
    // exactly one diagonal that contributes to one triangle. The diagonals are
    // consistent between neighboring hexahedrons so that triangles from
    // different hexahedrons have conformal connections that form planar
    // meshes.
    topology.NumberOfCells = hexTopology.NumberOfCells * 2;
    topology.NumberOfPoints = hexTopology.NumberOfPoints;

    // Make connections.
    connectArray.reserve(topology.NumberOfCells*CellType::NUM_POINTS);

    for (dax::Id cellIndex = 0; cellIndex < topology.NumberOfCells; cellIndex++)
      {
      dax::Tuple<dax::Id,CellType::NUM_POINTS> pointConnections
          = TestTopology::GetCellConnectionsImpl(topology, cellIndex);

      connectArray.push_back(pointConnections[0]);
      connectArray.push_back(pointConnections[1]);
      connectArray.push_back(pointConnections[2]);
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
  }

  static dax::Tuple<dax::Id,4> GetCellConnectionsImpl(
      const dax::exec::internal::TopologyUnstructured<
          dax::exec::CellQuadrilateral,TestExecutionAdapter> &,
      dax::Id cellIndex)
  {
    // Quadrilateral meshes are based on hex meshes.
    dax::Tuple<dax::Id,8> hexConnections =
        TestTopology::GetCellConnectionsImpl(TestTopology::GetCoreTopology(),
                                             cellIndex);
    dax::Tuple<dax::Id,4> quadConnections;
    quadConnections[0] = hexConnections[0];
    quadConnections[1] = hexConnections[2];
    quadConnections[2] = hexConnections[6];
    quadConnections[3] = hexConnections[4];

    return quadConnections;
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellQuadrilateral, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the hexahedron topology.
    dax::exec::internal::TopologyUnstructured<
        dax::exec::CellHexahedron,TestExecutionAdapter> hexTopology;
    std::vector<dax::Id> hexConnections;
    BuildTopology(hexTopology, coordArray, hexConnections);

    typedef dax::exec::CellQuadrilateral CellType;

    // Our mesh will have one quadrilateral for each hexahedron that cuts the
    // hexahedron diagonally in the x-y direction.

    topology.NumberOfCells = hexTopology.NumberOfCells;
    topology.NumberOfPoints = hexTopology.NumberOfPoints;

    // Make connections.
    connectArray.reserve(topology.NumberOfCells*CellType::NUM_POINTS);

    for (dax::Id cellIndex = 0; cellIndex < topology.NumberOfCells; cellIndex++)
      {
      dax::Tuple<dax::Id,CellType::NUM_POINTS> pointConnections
          = TestTopology::GetCellConnectionsImpl(topology, cellIndex);

      connectArray.push_back(pointConnections[0]);
      connectArray.push_back(pointConnections[1]);
      connectArray.push_back(pointConnections[2]);
      connectArray.push_back(pointConnections[3]);
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
  }
};

template<class FunctionType>
void TryAllTopologyTypes(FunctionType function)
{
  std::cout << "--- dax::exec::CellVoxel" << std::endl;
  TestTopology<dax::exec::internal::TopologyUniform> voxelTopology;
  function(voxelTopology);

  std::cout << "--- dax::exec::CellHexahedron" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellHexahedron,TestExecutionAdapter> > hexahedronTopology;
  function(hexahedronTopology);

  std::cout << "--- dax::exec::CellTetrahedron" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellTetrahedron,TestExecutionAdapter> > tetrahedronTopology;
  function(tetrahedronTopology);

  std::cout << "--- dax::exec::CellWedge" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellWedge,TestExecutionAdapter> > wedgeTopology;
  function(wedgeTopology);

  std::cout << "--- dax::exec::CellTriangle" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellTriangle,TestExecutionAdapter> > triangleTopology;
  function(triangleTopology);

  std::cout << "--- dax::exec::CellQuadrilateral" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellQuadrilateral,TestExecutionAdapter> >
      quadrilateralTopology;
  function(quadrilateralTopology);
}

namespace detail {

template<class TopologyType, typename T>
void SetArraySize(const TopologyType &topology,
                  std::vector<T> &array,
                  dax::exec::internal::FieldAssociationPointTag) {
  array.resize(topology.GetNumberOfPoints());
}
template<class TopologyType, typename T>
void SetArraySize(const TopologyType &topology,
                  std::vector<T> &array,
                  dax::exec::internal::FieldAssociationCellTag) {
  array.resize(topology.GetNumberOfCells());
}

} // namespace details

template<template <typename, class> class FieldType,
         class TopologyGenType,
         typename T>
FieldType<T, typename TopologyGenType::ExecutionAdapter>
CreateField(const TopologyGenType &topology, std::vector<T> &array){
  typedef FieldType<T, typename TopologyGenType::ExecutionAdapter> FieldClass;
  detail::SetArraySize(topology, array, typename FieldClass::AssociationTag());
  return FieldClass(&array.front());
}

template<class TopologyGenType>
typename TopologyGenType::CellType CreateCell(const TopologyGenType &topology,
                                              dax::Id cellIndex)
{
  return typename TopologyGenType::CellType(topology.GetTopology(), cellIndex);
}

template<class TopologyGenType>
dax::exec::WorkMapField<typename TopologyGenType::CellType,
                        typename TopologyGenType::ExecutionAdapter>
CreateWorkMapField(const TopologyGenType &topology, dax::Id index) {
  typedef dax::exec::WorkMapField<typename TopologyGenType::CellType,
                                  typename TopologyGenType::ExecutionAdapter>
      WorkType;
  return WorkType(topology.GetTopology(),
                  index,
                  typename TopologyGenType::ExecutionAdapter());
}

template<class TopologyGenType>
dax::exec::WorkMapCell<typename TopologyGenType::CellType,
                       typename TopologyGenType::ExecutionAdapter>
CreateWorkMapCell(const TopologyGenType &topology, dax::Id cellIndex) {
  typedef dax::exec::WorkMapCell<typename TopologyGenType::CellType,
                                 typename TopologyGenType::ExecutionAdapter>
      WorkType;
  return WorkType(topology.GetTopology(),
                  cellIndex,
                  typename TopologyGenType::ExecutionAdapter());
}

}
}
} // namespace dax::exec::internal

#endif //__dax_exec_internal_TestingTopologyGenerator_h
