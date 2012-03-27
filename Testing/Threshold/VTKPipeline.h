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

#include <stdio.h>
#include <iostream>
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/worklet/Threshold.h>

#include <vector>

#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkThreshold.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkTrivialProducer.h>
#include <vtkUnstructuredGrid.h>


namespace
{
void PrintResults(int pipeline, double time, const char* name)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV, " << name <<" ,"
            << pipeline << "," << time << std::endl;
}

void RunVTKPipeline(const dax::cont::UniformGrid &dgrid, vtkImageData* grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  std::vector<dax::Scalar> elev(dgrid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> elevHandle(elev.begin(),elev.end());

  //use dax to compute the elevation
  dax::cont::worklet::Elevation(dgrid, dgrid.GetPoints(), elevHandle);

  //now that the results are back on the cpu, do threshold with VTK
  vtkSmartPointer<vtkFloatArray> vtkElevationPoints = vtkSmartPointer<vtkFloatArray>::New();
  vtkElevationPoints->SetName("Elevation");
  vtkElevationPoints->SetVoidArray(&elev[0],elev.size(),1);
  grid->GetPointData()->AddArray(vtkElevationPoints);

  vtkNew<vtkTrivialProducer> producer;
  producer->SetOutput(grid);
  producer->Update();

  vtkNew<vtkThreshold> threshold;
  threshold->SetInputConnection(producer->GetOutputPort());
  threshold->SetPointsDataTypeToFloat();
  threshold->AllScalarsOn();
  threshold->ThresholdBetween(0, 50);
  threshold->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,"Elevation");

  //call once to have the threshold pipeline have proper mtimes so the real threshold
  //is as fast as possible.
  threshold->Update();
  threshold->ThresholdBetween(0,100);

  Timer timer;
  threshold->Update();
  double time = timer.elapsed();

  vtkSmartPointer<vtkUnstructuredGrid> out = threshold->GetOutput();
  
  std::cout << "original GetNumberOfCells: " << dgrid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << out->GetNumberOfCells() << std::endl;

  std::cout << "original GetNumberOfPoints: " << dgrid.GetNumberOfPoints() << std::endl;
  std::cout << "threshold GetNumberOfPoints: " << out->GetNumberOfPoints() << std::endl;
  PrintResults(1, time, "VTK");
}

} // Anonymous namespace

