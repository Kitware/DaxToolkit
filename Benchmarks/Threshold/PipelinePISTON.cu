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

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_CUDA
#define BOOST_SP_DISABLE_THREADS

#include <dax/cuda/cont/DeviceAdapterCuda.h>
#include <vtkImageData.h>
#include <vtkNew.h>

#include "PipelinePISTON.h"
#include "PISTONPipeline.h"

void RunPipelinePISTON(int pipeline, const dax::cont::UniformGrid<>& dgrid, vtkImageData* grid)
{
  RunPISTONPipeline(dgrid,grid);
}

#include "ArgumentsParser.h"


//create a dax and vtk image structure of the same size
dax::cont::UniformGrid<> CreateStructures(vtkImageData *grid, dax::Id dim)
{

  grid->SetOrigin(0.0, 0.0, 0.0);
  grid->SetSpacing(1.0, 1.0, 1.0);
  grid->SetExtent(0, dim-1,0, dim-1,0, dim-1);

  dax::cont::UniformGrid<> dgrid;
  dgrid.SetOrigin(dax::make_Vector3(0.0, 0.0, 0.0));
  dgrid.SetSpacing(dax::make_Vector3(1.0, 1.0, 1.0));
  dgrid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(dim-1, dim-1, dim-1));
  return dgrid;
}

int main(int argc, char* argv[])
  {
  dax::testing::ArgumentsParser parser;
  if (!parser.parseArguments(argc, argv))
    {
    return 1;
    }

  //init grid vars from parser
  const dax::Id MAX_SIZE = parser.problemSize();

  vtkNew<vtkImageData> grid;
  dax::cont::UniformGrid<> dgrid = CreateStructures(grid.GetPointer(),MAX_SIZE);

  int pipeline = parser.pipeline();
  std::cout << "Pipeline #" << pipeline << std::endl;

  RunPipelinePISTON(pipeline, dgrid, grid.GetPointer());

  return 0;
}
