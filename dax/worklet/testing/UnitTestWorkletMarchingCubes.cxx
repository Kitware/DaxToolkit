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
const dax::Id ISOVALUE = 70;

//-----------------------------------------------------------------------------
struct TestMarchingCubesWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > inGrid(DIM);
    dax::cont::UnstructuredGrid<dax::exec::CellTriangle> outGrid;

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

    //schedule marching cubes worklet generate step, saving
    //the coordinates into outGridCoordinates.
    typedef dax::Tuple<dax::Vector3,3> TriCoordinatesType;
    dax::cont::ArrayHandle<TriCoordinatesType> outGridCoordinates;

    std::cout << "Running Marching Cubes worklet" << std::endl;
    const dax::Scalar isoValue = ISOVALUE;

    try
      {
      typedef dax::cont::ScheduleGenerateTopology<dax::worklet::MarchingCubesTopology> ScheduleGT;
      typedef typename ScheduleGT::ClassifyResultType  ClassifyResultType;

      //construct the scheduler that will execute all the worklets
      dax::cont::Scheduler<> scheduler;

      //construct the two worklets that will be used to do the marching cubes
      dax::worklet::MarchingCubesClassify classifyWorklet(isoValue);
      dax::worklet::MarchingCubesTopology generateWorklet(isoValue);


      //run the first step
      ClassifyResultType classification; //array handle for the first step classification
      scheduler.Invoke(classifyWorklet, inGrid.GetRealGrid(),
                       fieldHandle, classification);

      //construct the topology generation worklet
      ScheduleGT generate(classification,generateWorklet);
      generate.SetRemoveDuplicatePoints(false);

      //run the second step
      scheduler.Invoke(generate,
                       inGrid.GetRealGrid(), outGrid, fieldHandle,
                       inGrid->GetPointCoordinates(),
                       outGridCoordinates);

      //we can't save the coordiantes to the grid just yet.
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
    DAX_TEST_ASSERT(outGrid.GetNumberOfCells()==outGridCoordinates.GetNumberOfValues(),
                    "Incorrect number of points in the output grid");

    }
};


//-----------------------------------------------------------------------------
void TestMarchingCubes()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestMarchingCubesWorklet());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMarchingCubes(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestMarchingCubes);
}
