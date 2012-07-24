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


// These header files help tease out when the default template arguments to
// ArrayHandle are inappropriately used.
#include <dax/cont/internal/ArrayContainerControlError.h>
#include <dax/cont/internal/DeviceAdapterError.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/internal/TestingGridGenerator.h>
#include <dax/cont/internal/Testing.h>

#include <dax/cont/worklet/Threshold.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 64;
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
    if (!isValid)
      {
      std::cout << "*** Encountered bad value." << std::endl;
      std::cout << std::distance(begin,iter) << ":";
      dax::cont::VectorForEach(vector, PrintScalarValue);
      std::cout << std::endl;
      break;
      }
    }
  }

//-----------------------------------------------------------------------------
struct TestThresholdWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    this->GridThreshold<GridType,GridType>();
    }

  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    this->GridThreshold<dax::cont::UniformGrid<>,
        dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> >();
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridThreshold() const
    {
    dax::cont::internal::TestGrid<InGridType> grid(DIM);
    dax::cont::internal::TestGrid<OutGridType>grid2(0);

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 coordinates = grid->ComputePointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle(&field[0],
                                                    &field[grid->GetNumberOfPoints()]);

    //unkown size
    dax::cont::ArrayHandle<dax::Scalar> resultHandle;

    std::cout << "Running Threshold worklet" << std::endl;
    dax::Scalar min = MIN_THRESHOLD;
    dax::Scalar max = MAX_THRESHOLD;

    try
      {
      dax::cont::worklet::Threshold(grid,grid2,min,max,fieldHandle,resultHandle);
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }

    DAX_TEST_ASSERT(resultHandle.GetNumberOfValues()==grid2->GetNumberOfPoints(),
                    "Incorrect number of points in the result array");

    CheckValues(resultHandle.GetIteratorConstControlBegin(),
                resultHandle.GetIteratorConstControlEnd());
    //test max < min.
    }
};


//-----------------------------------------------------------------------------
void TestThreshold()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestThresholdWorklet());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletThreshold(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestThreshold);
}
