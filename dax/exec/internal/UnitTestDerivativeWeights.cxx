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

#include <dax/exec/internal/DerivativeWeights.h>

#include <dax/internal/Testing.h>

namespace {

static void TestWeightOnVertex(dax::Vector3 weight,
                               dax::Vector3 derivativePCoord,
                               dax::Vector3 vertexPCoord)
{
  dax::Vector3 signs = 2.0*vertexPCoord - dax::make_Vector3(1.0, 1.0, 1.0);

  if (vertexPCoord == derivativePCoord)
    {
    DAX_TEST_ASSERT(weight == signs, "Bad Vertex Weight");
    }
  else if (   (vertexPCoord[0] != derivativePCoord[0])
           && (vertexPCoord[1] == derivativePCoord[1])
           && (vertexPCoord[2] == derivativePCoord[2]) )
    {
    DAX_TEST_ASSERT(weight == signs*dax::make_Vector3(1.0, 0.0, 0.0),
                    "Bad Vertex Weight");
    }
  else if (   (vertexPCoord[0] == derivativePCoord[0])
           && (vertexPCoord[1] != derivativePCoord[1])
           && (vertexPCoord[2] == derivativePCoord[2]) )
    {
    DAX_TEST_ASSERT(weight == signs*dax::make_Vector3(0.0, 1.0, 0.0),
                    "Bad Vertex Weight");
    }
  else if (   (vertexPCoord[0] == derivativePCoord[0])
           && (vertexPCoord[1] == derivativePCoord[1])
           && (vertexPCoord[2] != derivativePCoord[2]) )
    {
    DAX_TEST_ASSERT(weight == signs*dax::make_Vector3(0.0, 0.0, 1.0),
                    "Bad Vertex Weight");
    }
  else
    {
    DAX_TEST_ASSERT(weight == dax::make_Vector3(0.0, 0.0, 0.0),
                    "Bad Vertex Weight");
    }
}

static void TestWeightInMiddle(dax::Scalar weight,
                               dax::Scalar vertexPCoord)
{
  if (vertexPCoord < 0.5)
    {
    DAX_TEST_ASSERT(weight == -0.25, "Bad middle weight");
    }
  else
    {
    DAX_TEST_ASSERT(weight == 0.25, "Bad middle weight");
    }
}

static void TestDerivativeWeightsVoxel()
{
  std::cout << "In TestDerivativeWeightsVoxel" << std::endl;

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

  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 pcoords = cellVertexToParametricCoords[vertexIndex];

    dax::Tuple<dax::Vector3,8> weights =
        dax::exec::internal::derivativeWeightsVoxel(pcoords);

    for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
      {
      dax::Vector3 vertexPCoords = cellVertexToParametricCoords[weightIndex];

      TestWeightOnVertex(weights[weightIndex], pcoords, vertexPCoords);
      }
    }

  // Check for Derivative at middle.
  dax::Tuple<dax::Vector3,8> weights =
      dax::exec::internal::derivativeWeightsVoxel(dax::make_Vector3(0.5,0.5,0.5));
  for (dax::Id weightIndex = 0; weightIndex < 8; weightIndex++)
    {
    dax::Vector3 vertexPCoords = cellVertexToParametricCoords[weightIndex];

    TestWeightInMiddle(weights[weightIndex][0], vertexPCoords[0]);
    TestWeightInMiddle(weights[weightIndex][1], vertexPCoords[1]);
    TestWeightInMiddle(weights[weightIndex][2], vertexPCoords[2]);
    }
}

void TestDerivativeWeights()
{
  TestDerivativeWeightsVoxel();
}

} // Anonymous namespace

int UnitTestDerivativeWeights(int, char *[])
{
  return dax::internal::Testing::Run(TestDerivativeWeights);
}
