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

#include <dax/worklet/Sine.h>

#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
struct TestSineWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
  {
  dax::cont::internal::TestGrid<GridType> grid(DIM);

  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid->ComputePointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar,
                        dax::cont::ArrayContainerControlTagBasic,
                        dax::cont::DeviceAdapterTagSerial> fieldHandle =
      dax::cont::make_ArrayHandle(field,
                        dax::cont::ArrayContainerControlTagBasic(),
                        dax::cont::DeviceAdapterTagSerial());

  dax::cont::ArrayHandle<dax::Scalar,
                        dax::cont::ArrayContainerControlTagBasic,
                        dax::cont::DeviceAdapterTagSerial> sineHandle;

  std::cout << "Running Sine worklet" << std::endl;
  dax::cont::Scheduler<> scheduler;
  scheduler.Invoke(dax::worklet::Sine(),fieldHandle, sineHandle);

  std::cout << "Checking result" << std::endl;
  std::vector<dax::Scalar> sine(grid->GetNumberOfPoints());
  sineHandle.CopyInto(sine.begin());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar sineValue = sine[pointIndex];
    dax::Scalar sineTrue = sinf(field[pointIndex]);
    DAX_TEST_ASSERT(test_equal(sineValue, sineTrue),
                    "Got bad sine");
    }

  std::cout << "Running Sine worklet on constant ArrayHandle" << std::endl;
  dax::cont::ArrayHandleConstantValue<dax::Scalar> constantHandle(5.034,
                                                    grid->GetNumberOfPoints());
  scheduler.Invoke(dax::worklet::Sine(), constantHandle, sineHandle);


  std::cout << "Checking result of constant handle" << std::endl;
  sineHandle.CopyInto(sine.begin());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar sineValue = sine[pointIndex];
    dax::Scalar sineTrue = sinf(5.034);
    DAX_TEST_ASSERT(test_equal(sineValue, sineTrue),
                    "Got bad sine");
    }

  std::cout << "Running Sine worklet on constant value" << std::endl;
  scheduler.Invoke(dax::worklet::Sine(), 5.034, sineHandle);

  std::cout << "Checking result of constant value" << std::endl;
  sineHandle.CopyInto(sine.begin());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar sineValue = sine[pointIndex];
    dax::Scalar sineTrue = sinf(5.034);
    DAX_TEST_ASSERT(test_equal(sineValue, sineTrue),
                    "Got bad sine");
    }

  }
};

//-----------------------------------------------------------------------------
void TestSine()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestSineWorklet());
  }

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletSine(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestSine);
}
