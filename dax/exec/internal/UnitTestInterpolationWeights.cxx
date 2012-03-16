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

#include <dax/exec/internal/InterpolationWeights.h>

#include <dax/internal/Testing.h>

namespace {

static void TestInterpolationWeightsVoxel()
{
  std::cout << "In TestInterpolationWeightsVoxel" << std::endl;

  const dax::Vector3 cellVertexToParametricCoords[8] = {
    dax::make_Vector3(0, 0, 0),
    dax::make_Vector3(1, 0, 0),
    dax::make_Vector3(1, 1, 0),
    dax::make_Vector3(0, 1, 0),
    dax::make_Vector3(0, 0, 1),
    dax::make_Vector3(1, 0, 1),
    dax::make_Vector3(1, 1, 1),
    dax::make_Vector3(0, 1, 1)
  };

  // Check interpolation at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 pcoords = cellVertexToParametricCoords[vertexIndex];

    dax::Scalar weights[8];
    dax::exec::internal::interpolationWeightsVoxel(pcoords, weights);

    for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
      {
      if (weightIndex == vertexIndex)
        {
        DAX_TEST_ASSERT(weights[weightIndex] == 1.0,
                        "Got bad interpolation weight");
        }
      else
        {
        DAX_TEST_ASSERT(weights[weightIndex] == 0.0,
                        "Got bad interpolation weight");
        }
      }
    }

  // Check for interpolation at middle.
  dax::Scalar weights[8];
  dax::exec::internal::interpolationWeightsVoxel(dax::make_Vector3(0.5,0.5,0.5),
                                                 weights);
  for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
    {
    DAX_TEST_ASSERT(weights[weightIndex] == 0.125,
                    "Got bad interpolation weight");
    }
}

void TestInterpolationWeights()
{
  TestInterpolationWeightsVoxel();
}

}

int UnitTestInterpolationWeights(int, char *[])
{
  return dax::internal::Testing::Run(TestInterpolationWeights);
}
