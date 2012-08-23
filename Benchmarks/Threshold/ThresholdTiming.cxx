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

// This is how I really want the test to behave. I want one program that
// switches between devices. Unfortunately, that is not currently possible
// because both the OpenMP and CUDA devices use thrust as part of its
// implementation. The compiling of thrust with different flags causes
// confusing with the linker and I don't see an easy way to isolate the
// instances. Thus, this file currently languishes.


#include "ArgumentsParser.h"

#include "ThresholdTimingConfig.h"

#include "PipelineSerial.h"

#ifdef DAX_ENABLE_OPENMP
#include "PipelineOpenMP.h"
#endif

#ifdef DAX_ENABLE_CUDA
#include "PipelineCuda.h"
#endif

#include <iostream>

namespace {

dax::cont::UniformGrid CreateInputStructure(dax::Id dim)
{
  dax::cont::UniformGrid grid;
  grid.SetOrigin(dax::make_Vector3(0.0, 0.0, 0.0));
  grid.SetSpacing(dax::make_Vector3(1.0, 1.0, 1.0));
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(dim-1, dim-1, dim-1));
  return grid;
}

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

  dax::cont::UniformGrid grid = CreateInputStructure(MAX_SIZE);

  int pipeline = parser.pipeline();
  std::cout << "Pipeline #" << pipeline << std::endl;

  switch (parser.device())
    {
    case dax::testing::ArgumentsParser::DEVICE_ALL:
      RunPipelineSerial(pipeline, grid);
#ifdef DAX_ENABLE_OPENMP
      RunPipelineOpenMP(pipeline, grid);
#endif
#ifdef DAX_ENABLE_CUDA
      RunPipelineCuda(pipeline, grid);
#endif
      break;
    case dax::testing::ArgumentsParser::DEVICE_SERIAL:
      RunPipelineSerial(pipeline, grid);
      break;
    case dax::testing::ArgumentsParser::DEVICE_OPENMP:
#ifdef DAX_ENABLE_OPENMP
      RunPipelineOpenMP(pipeline, grid);
      break;
#else
      std::cout << "OpenMP device not available." << std::endl;
      return 1;
#endif
    case dax::testing::ArgumentsParser::DEVICE_CUDA:
#ifdef DAX_ENABLE_CUDA
      RunPipelineCuda(pipeline, grid);
      break;
#else
      std::cout << "Cuda device not available." << std::endl;
      return 1;
#endif
    }

  return 0;
}
