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

#include <dax/cont/Scheduler.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/worklet/Magnitude.worklet>

#include <vector>

#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkThreshold.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>

#include <vector>

#define SPACE thrust::detail::default_device_space_tag
#include <piston/piston_math.h>
#include <piston/choose_container.h>
#include <piston/image3d.h>
#include <piston/vtk_image3d.h>
#include <piston/threshold_geometry.h>

namespace
{
void PrintResults(int pipeline, double time, const char* name)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV, " << name <<" ,"
            << pipeline << "," << time << std::endl;
}

void RunPISTONPipeline(const dax::cont::UniformGrid<> &dgrid, vtkImageData* grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  std::vector<dax::Scalar> elev(dgrid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> elevHandle = dax::cont::make_ArrayHandle(elev);

  //use dax to compute the magnitude
  dax::cont::Scheduler<> scheduler;
  scheduler.Invoke(dax::worklet::Magnitude(),
            dgrid.GetPointCoordinates(),
            elevHandle);

  //return results to host
  elevHandle.CopyInto(elev.begin());

  //now that the results are back on the cpu, do threshold with VTK
  vtkSmartPointer<vtkFloatArray> vtkElevationPoints = vtkSmartPointer<vtkFloatArray>::New();
  vtkElevationPoints->SetName("Elevation");
  vtkElevationPoints->SetVoidArray(&elev[0],elev.size(),1);

  grid->GetPointData()->SetScalars(vtkElevationPoints); //piston on works on active scalars
  piston::vtk_image3d<SPACE> image(grid);
  piston::threshold_geometry<piston::vtk_image3d<SPACE> > threshold(image,0,100);

  Timer timer;
  threshold();
  double time = timer.elapsed();

  std::cout << "original GetNumberOfCells: " << dgrid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << threshold.valid_cell_indices.size() << std::endl;
  PrintResults(1, time, "Piston");
}




} // Anonymous namespace

