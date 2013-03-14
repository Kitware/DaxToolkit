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

#include "ArgumentsParser.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/GenerateInterpolatedCells.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/MarchingCubes.h>

#include <iostream>
#include <vector>
#include <fstream>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace
{

dax::Scalar ISOVALUE = 5;

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

void RunDAXPipeline(const dax::cont::UniformGrid<> &grid, int pipeline)
{
  std::cout << "Running pipeline 1: Magnitude -> MarchingCubes" << std::endl;

  dax::cont::UnstructuredGrid<dax::CellTagTriangle> outGrid;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::Scheduler<> schedule;
  schedule.Invoke(dax::worklet::Magnitude(),
        grid.GetPointCoordinates(),
        intermediate1);

  dax::cont::Timer<> timer;

  //schedule marching cubes worklet generate step
  typedef dax::cont::GenerateInterpolatedCells<dax::worklet::MarchingCubesTopology> GenerateIC;
  typedef GenerateIC::ClassifyResultType  ClassifyResultType;

  dax::worklet::MarchingCubesClassify classifyWorklet(ISOVALUE);
  dax::worklet::MarchingCubesTopology generateWorklet(ISOVALUE);


  //run the first step
  ClassifyResultType classification; //array handle for the first step classification
  schedule.Invoke(classifyWorklet, grid,
                   intermediate1, classification);

  //construct the topology generation worklet
  GenerateIC generate(classification,generateWorklet);

  generate.SetRemoveDuplicatePoints(
                pipeline == dax::testing::ArgumentsParser::MARCHING_CUBES_REMOVE_DUPLICATES);

  //run the second step
  schedule.Invoke(generate,
                   grid, outGrid, intermediate1);

  double time = timer.GetElapsedTime();

  std::cout << "number of coordinates in: " << grid.GetNumberOfPoints() << std::endl;
  std::cout << "number of coordinates out: " << outGrid.GetNumberOfPoints() << std::endl;
  std::cout << "number of cells out: " << outGrid.GetNumberOfCells() << std::endl;
  PrintResults(1, time);

}


} // Anonymous namespace

