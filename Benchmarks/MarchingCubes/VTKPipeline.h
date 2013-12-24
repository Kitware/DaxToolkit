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

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/worklet/Magnitude.h>

#include <vector>

#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkTrivialProducer.h>
#include <vtkPolyData.h>


namespace
{

dax::Scalar ISOVALUE = 100;

void PrintResults(int pipeline, double time, const char* name)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV, " << name <<" ,"
            << pipeline << "," << time << std::endl;
}

void RunVTKPipeline(const dax::cont::UniformGrid<> &dgrid, vtkImageData* grid)
{
  std::cout << "Running pipeline 1: Elevation -> MarchingCubes" << std::endl;

  std::vector<dax::Scalar> elev(dgrid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> elevHandle;

  //use dax to compute the elevation
  dax::cont::DispatcherMapField< dax::worklet::Magnitude > dispatcher;
  dispatcher.Invoke( dgrid.GetPointCoordinates(), elevHandle);

  elevHandle.CopyInto(elev.begin());

  //now that the results are back on the cpu, do threshold with VTK
  vtkSmartPointer<vtkFloatArray> vtkElevationPoints = vtkSmartPointer<vtkFloatArray>::New();
  vtkElevationPoints->SetName("Elevation");
  vtkElevationPoints->SetVoidArray(&elev[0],elev.size(),1);
  grid->GetPointData()->AddArray(vtkElevationPoints);
  grid->GetPointData()->SetActiveScalars("Elevation");

  vtkNew<vtkTrivialProducer> producer;
  producer->SetOutput(grid);
  producer->Update();

  vtkNew<vtkMarchingCubes> marching;
  marching->SetInputConnection(producer->GetOutputPort());

  dax::cont::Timer<> timer;
  marching->ComputeNormalsOff();
  marching->ComputeGradientsOff();
  marching->ComputeScalarsOn();
  marching->SetNumberOfContours(1);
  marching->SetValue(0, ISOVALUE);


  marching->Update();
  double time = timer.GetElapsedTime();

  vtkSmartPointer<vtkPolyData> out = marching->GetOutput();

  std::cout << "number of coordinates in: " << grid->GetNumberOfPoints() << std::endl;
  std::cout << "number of coordinates out: " << out->GetNumberOfPoints() << std::endl;
  std::cout << "number of cells out: " << out->GetNumberOfCells() << std::endl;

  PrintResults(1, time, "VTK");
}

} // Anonymous namespace

