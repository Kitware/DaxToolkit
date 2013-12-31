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

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/Threshold.h>
#include <dax/worklet/testing/VerifyThresholdTopology.h>

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/DispatcherGenerateTopology.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/testing/Testing.h>

#include <iostream>
#include <vector>

namespace {
const dax::Id DIM = 26;
const dax::Id MIN_THRESHOLD = 70;
const dax::Id MAX_THRESHOLD = 82;


class CheckValid {
public:
  CheckValid() : Valid(true) { }
  DAX_CONT_EXPORT operator bool() { return this->Valid; }
  DAX_CONT_EXPORT void operator()(dax::Scalar value) {
    if ((value < MIN_THRESHOLD) || (value > MAX_THRESHOLD)) {
      this->Valid = false; }
    }
private:
  bool Valid;
};


template<class IteratorType>
DAX_CONT_EXPORT
void CheckValues(IteratorType begin, IteratorType end)
  {
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  CheckValid isValid;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    VectorType vector = *iter;
    dax::cont::VectorForEach(vector, isValid);
    DAX_TEST_ASSERT(isValid, "Encountered bad value.");
    }
  }

template<typename T, class Container, class Device>
DAX_CONT_EXPORT
void CheckValues(dax::cont::ArrayHandle<T,Container,Device> arrayHandle)
{
  CheckValues(arrayHandle.GetPortalConstControl().GetIteratorBegin(),
              arrayHandle.GetPortalConstControl().GetIteratorEnd());
}


template<class InGridGeneratorType,
         class ConnectionsPortalType,
         class CoordinatesPortalType,
         class CellTag>
DAX_CONT_EXPORT
void CheckConnections(const InGridGeneratorType &inGridGenerator,
                      const std::vector<dax::Scalar> &inField,
                      ConnectionsPortalType connectionsPortal,
                      CoordinatesPortalType coordinatesPortal,
                      CellTag)
{
  dax::Id outConnectionIndex = 0;

  for (dax::Id inCellIndex = 0;
       inCellIndex < inGridGenerator->GetNumberOfCells();
       inCellIndex++)
    {
    dax::cont::testing::CellConnections<CellTag> inPointIndices =
                              inGridGenerator.GetCellConnections(inCellIndex);

    CheckValid isValid;
    for (int vertexIndex = 0;
         vertexIndex < inPointIndices.NUM_VERTICES;
         vertexIndex++)
      {
      isValid( inField[inPointIndices[vertexIndex]] );
      }

    if (!isValid)
      {
      // Cell isn't one that passed the threshold so don't verify it
      continue;
      }

    // If we are here, this cell should have been passed and the next
    // connections should match coordinates.
    DAX_TEST_ASSERT(outConnectionIndex < connectionsPortal.GetNumberOfValues(),
                    "Output does not have enough cells.");

    dax::cont::testing::CellCoordinates<CellTag> inCoordinates =
        inGridGenerator.GetCellVertexCoordinates(inCellIndex);
    for (int vertexIndex = 0;
         vertexIndex < inPointIndices.NUM_VERTICES;
         vertexIndex++)
      {
      dax::Vector3 outCoordinates =
          coordinatesPortal.Get(connectionsPortal.Get(outConnectionIndex));
      outConnectionIndex++;

      DAX_TEST_ASSERT(test_equal(inCoordinates[vertexIndex], outCoordinates),
                      "Got bad coordinates in output.");
      }
    }

  DAX_TEST_ASSERT(outConnectionIndex == connectionsPortal.GetNumberOfValues(),
                  "Output has too many cells.");
}

//-----------------------------------------------------------------------------
struct TestThresholdWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  DAX_CONT_EXPORT
  void operator()(const GridType&) const
    {
    dax::cont::testing::TestGrid<GridType> in(DIM);
    GridType out;

    this->GridThreshold(in,out);
    }

  //----------------------------------------------------------------------------
  DAX_CONT_EXPORT
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::testing::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::CellTagHexahedron> out;

    this->GridThreshold(in,out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  DAX_CONT_EXPORT
  void GridThreshold(
      const dax::cont::testing::TestGrid<InGridType> &inGridGenerator,
      OutGridType& outGrid) const
    {
    const InGridType inGrid = inGridGenerator.GetRealGrid();

    typedef typename InGridType::CellTag InCellTag;
    typedef typename OutGridType::CellTag OutCellTag;

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    std::vector<dax::Scalar> field(inGrid.GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < inGrid.GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.ComputePointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle =
        dax::cont::make_ArrayHandle(field);

    //unkown size
    dax::cont::ArrayHandle<dax::Scalar> resultHandle;

    std::cout << "Running Threshold worklet" << std::endl;
    dax::Scalar min = MIN_THRESHOLD;
    dax::Scalar max = MAX_THRESHOLD;

    try
      {
      typedef dax::cont::DispatcherGenerateTopology<
            dax::worklet::testing::VerifyThresholdTopology > DispatcherGT;
      typedef typename DispatcherGT::CountHandleType  CountHandleType;


      typedef dax::worklet::ThresholdCount< dax::Scalar> CountWorklet;
      dax::cont::DispatcherMapCell< CountWorklet > classifyDispatcher(
                                                    CountWorklet(min,max) );


      CountHandleType count;
      classifyDispatcher.Invoke(inGrid, fieldHandle, count);

      //construct the topology generation worklet
      DispatcherGT dispatcherTopo(count);

      dispatcherTopo.Invoke( inGrid, outGrid, 4.0f );

      //request to also compact the topology
      dispatcherTopo.CompactPointField(fieldHandle,resultHandle);
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }

    DAX_TEST_ASSERT(resultHandle.GetNumberOfValues() ==
                    outGrid.GetNumberOfPoints(),
                    "Incorrect number of points in the result array");
    CheckValues(resultHandle);

    DAX_TEST_ASSERT(outGrid.GetCellConnections().GetNumberOfValues() ==
                    ( outGrid.GetNumberOfCells()
                      * dax::CellTraits<OutCellTag>::NUM_VERTICES ),
                    "Size of cell connections array incorrect.");
    CheckConnections(inGridGenerator,
                     field,
                     outGrid.GetCellConnections().GetPortalConstControl(),
                     outGrid.GetPointCoordinates().GetPortalConstControl(),
                     InCellTag());
    }
};


//-----------------------------------------------------------------------------
static void TestThreshold()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(TestThresholdWorklet());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletThreshold(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestThreshold);
}
