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

#include <dax/worklet/MarchingCubes.worklet>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/GenerateInterpolatedCells.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/testing/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id ISOVALUE = 70;

//-----------------------------------------------------------------------------
struct TestMarchingCubesWorklet
{
  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    typedef dax::CellTagTriangle CellType;

    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > inGrid(DIM);
    dax::cont::UnstructuredGrid<CellType> outGrid;

    std::cout << "running " << std::endl;

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    dax::Id numPoints = inGrid->GetNumberOfPoints();
    std::vector<dax::Scalar> field(numPoints);
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.GetPointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle =
        dax::cont::make_ArrayHandle(field);

    std::cout << "Running Marching Cubes worklet" << std::endl;
    const dax::Scalar isoValue = ISOVALUE;

    try
      {
      typedef dax::cont::GenerateInterpolatedCells<dax::worklet::MarchingCubesTopology> GenerateIC;
      typedef GenerateIC::ClassifyResultType  ClassifyResultType;

      //construct the scheduler that will execute all the worklets
      dax::cont::Scheduler<> scheduler;

      //construct the two worklets that will be used to do the marching cubes
      dax::worklet::MarchingCubesClassify classifyWorklet(isoValue);
      dax::worklet::MarchingCubesTopology generateWorklet(isoValue);


      //run the first step
      ClassifyResultType classification; //array handle for the first step classification
      scheduler.Invoke(classifyWorklet,
                       inGrid.GetRealGrid(),
                       fieldHandle,
                       classification);

      //construct the topology generation worklet
      GenerateIC generate(classification,generateWorklet);
      generate.SetRemoveDuplicatePoints(false);
      //so we can use the classification again
      generate.SetReleaseClassification(false);

      //run the second step
      scheduler.Invoke(generate,
                       inGrid.GetRealGrid(),
                       outGrid,
                       fieldHandle);

    const dax::Id valid_num_points =
          dax::CellTraits<CellType>::NUM_VERTICES * outGrid.GetNumberOfCells();
    DAX_TEST_ASSERT(valid_num_points==outGrid.GetNumberOfPoints(),
                    "Incorrect number of points in the output grid when not merging");

    generate.SetRemoveDuplicatePoints(true);
    //run the second step again with point merging

    dax::cont::UnstructuredGrid<CellType> secondOutGrid;
    scheduler.Invoke(generate,
                     inGrid.GetRealGrid(),
                     secondOutGrid,
                     fieldHandle);

    //45 is the number I got from running the algorithm by hand and seeing
    //the number of unique points
    const dax::Id NumberOfUniquePoints = 45;
    DAX_TEST_ASSERT(NumberOfUniquePoints == secondOutGrid.GetNumberOfPoints() &&
                    NumberOfUniquePoints != valid_num_points,
        "We didn't merge to the correct number of points");
    }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
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
