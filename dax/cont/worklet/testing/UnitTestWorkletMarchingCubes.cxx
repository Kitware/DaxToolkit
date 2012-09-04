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
#include <dax/cont/worklet/MarchingCubes.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UnstructuredGrid.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id ISO_VALUE = 70;

//-----------------------------------------------------------------------------
struct TestMarchingCubesWorklet
{
  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::exec::CellTriangle> out;

    this->GridMarchingCubes(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridMarchingCubes(const InGridType& inGrid, OutGridType& outGrid) const
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

    std::cout << "Running MarchingCubes worklet" << std::endl;
    dax::Scalar isoValue = ISO_VALUE;

    dax::cont::worklet::MarchingCubes(inGrid,
                                      outGrid,
                                      isoValue,
                                      fieldHandle);

    //how will we verify this now? we need some way to save and encode the
    //proper coordinates and compare that here instead
    DAX_TEST_ASSERT(outGrid.GetNumberOfPoints()!=inGrid.GetNumberOfPoints(),
                    "Incorrect number of points in the result array");
    }
};


//-----------------------------------------------------------------------------
void TestMarchingCubes()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestMarchingCubesWorklet(),
                      dax::cont::internal::GridTesting::TypeCheckUniformGrid());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMarchingCubes(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestMarchingCubes);
}
