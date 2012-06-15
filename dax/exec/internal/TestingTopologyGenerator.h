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

private:
  static void BuildTopology(dax::exec::internal::TopologyUniform &topology,
                            std::vector<dax::Vector3> &/*coords*/,
                            std::vector<dax::Id> &/*connections*/)
  {
    topology.Origin = dax::make_Vector3(1.0, -0.5, 13.0);
    topology.Spacing = dax::make_Vector3(2.5, 6.25, 1.0);
    topology.Extent.Min = dax::make_Id3(5, -2, -7);
    topology.Extent.Max = dax::make_Id3(20, 4, 10);
  }

  static void BuildTopology(dax::exec::internal::TopologyUnstructured
                            <dax::exec::CellHexahedron, TestExecutionAdapter>
                            &topology,
                            std::vector<dax::Vector3> &coordArray,
                            std::vector<dax::Id> &connectArray)
  {
    // Base this topology on the uniform topology.
    dax::exec::internal::TopologyUniform uniformTopology;
    TestTopology::BuildTopology(uniformTopology, coordArray, connectArray);

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

    const dax::Extent3 extent = uniformTopology.Extent;
    for(dax::Id flatCellIndex = 0;
        flatCellIndex < topology.NumberOfCells;
        flatCellIndex++)
      {
      dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(flatCellIndex, extent);
      for(dax::Id relativePointIndex = 0;
          relativePointIndex < CellType::NUM_POINTS;
          relativePointIndex++)
        {
        dax::Id3 ijkPoint = ijkCell+cellVertexToPointIndex[relativePointIndex];

        dax::Id pointIndex = index3ToFlatIndex(ijkPoint,extent);
        connectArray.push_back(pointIndex);
        }
      }
    DAX_TEST_ASSERT(dax::Id(connectArray.size())
                    == topology.NumberOfCells*CellType::NUM_POINTS,
                    "Bad connection array size.");

    topology.CellConnections = &connectArray.front();
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

    std::vector<dax::Id>::iterator hexCellIter = hexConnections.begin();
    for (dax::Id cellIndex = 0;
         cellIndex < hexTopology.NumberOfCells;
         cellIndex++)
      {
      dax::Tuple<dax::Id,8> hex;
      hex[0] = *(hexCellIter++);
      hex[1] = *(hexCellIter++);
      hex[2] = *(hexCellIter++);
      hex[3] = *(hexCellIter++);
      hex[4] = *(hexCellIter++);
      hex[5] = *(hexCellIter++);
      hex[6] = *(hexCellIter++);
      hex[7] = *(hexCellIter++);

      // Triangle 1.
      connectArray.push_back(hex[0]);
      connectArray.push_back(hex[5]);
      connectArray.push_back(hex[7]);

      // Triangle 2.
      connectArray.push_back(hex[6]);
      connectArray.push_back(hex[3]);
      connectArray.push_back(hex[1]);
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

  std::cout << "--- dax::exec::CellTriangle" << std::endl;
  TestTopology<dax::exec::internal::TopologyUnstructured
      <dax::exec::CellTriangle,TestExecutionAdapter> > triangleTopology;
  function(triangleTopology);
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
