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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

#include <dax/worklet/Threshold.worklet>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/CellTag.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/ScheduleGenerateTopology.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/testing/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id MIN_THRESHOLD = 70;
const dax::Id MAX_THRESHOLD = 82;


class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < MIN_THRESHOLD) || (value > MAX_THRESHOLD)) {
      this->Valid = false; }
    }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
  {
  std::cout << " " << value;
  }


template<class IteratorType>
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
void CheckValues(dax::cont::ArrayHandle<T,Container,Device> arrayHandle)
{
  CheckValues(arrayHandle.GetPortalConstControl().GetIteratorBegin(),
              arrayHandle.GetPortalConstControl().GetIteratorEnd());
}


class TestThresholdTopology : public dax::exec::WorkletGenerateTopology
{
public:
  typedef void ControlSignature(Topology, Topology(Out),Field(In));
  typedef void ExecutionSignature(Topology::PointIds(_1),
                                  Topology::PointIds(_2),
                                  _3,
                                  VisitIndex);

  template<typename InputCellTag, typename OutputCellTag, typename T>
  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<InputCellTag> &inVertices,
                  dax::exec::CellVertices<OutputCellTag> &outVertices,
                  const T&,
                  const dax::Id& visit_index) const
  {
    DAX_TEST_ASSERT(visit_index==0, "Encountered bad visit index value.");
    outVertices.SetFromTuple(inVertices.GetAsTuple());
  }
};

//-----------------------------------------------------------------------------
struct TestThresholdWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::internal::TestGrid<GridType> in(DIM);
    GridType out;

    this->GridThreshold(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::CellTagHexahedron> out;

    this->GridThreshold(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridThreshold(const InGridType& inGrid, OutGridType& outGrid) const
    {
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
      typedef dax::cont::ScheduleGenerateTopology<TestThresholdTopology> ScheduleGT;
      typedef typename ScheduleGT::ClassifyResultType  ClassifyResultType;
      typedef dax::worklet::ThresholdClassify<dax::Scalar> ThresholdClassifyType;

      dax::cont::Scheduler<> scheduler;
      ClassifyResultType classification;
      scheduler.Invoke(ThresholdClassifyType(min,max),
                inGrid, fieldHandle, classification);

      //construct the topology generation worklet
      ScheduleGT generateTopo(classification);

      //schedule it, and verify we can handle more than 2 parameter generate
      //topology worklets
      scheduler.Invoke(generateTopo,inGrid, outGrid, 4.0f);

      //request to also compact the topology
      generateTopo.CompactPointField(fieldHandle,resultHandle);
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }

    CheckValues(resultHandle);
    DAX_TEST_ASSERT(resultHandle.GetNumberOfValues()==outGrid.GetNumberOfPoints(),
                    "Incorrect number of points in the result array");
    }
};


//-----------------------------------------------------------------------------
void TestThreshold()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestThresholdWorklet(),
                     dax::cont::internal::GridTesting::TypeCheckUniformGrid());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletThreshold(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestThreshold);
}
