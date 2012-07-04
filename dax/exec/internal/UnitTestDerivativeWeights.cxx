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

#include <dax/exec/ParametricCoordinates.h>

#include <dax/internal/Testing.h>

namespace {

static void TestWeightOnVertex(dax::Vector3 weight,
                               dax::Vector3 derivativePCoord,
                               dax::Vector3 vertexPCoord,
                               int dimensions)
{
  dax::Vector3 signs = 2.0*vertexPCoord - dax::make_Vector3(1.0, 1.0, 1.0);
  if (dimensions < 3) { signs[2] = 0.0; }

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
                               dax::Scalar vertexPCoord,
                               int dimensions)
{
  dax::Scalar expectedWeight = 0.0;
  switch (dimensions)
    {
    case 3:  expectedWeight = 0.25;  break;
    case 2:  expectedWeight = 0.5;   break;
    default: DAX_TEST_FAIL("Unknown dimensions.");
    }

  if (vertexPCoord < 0.5)
    {
    DAX_TEST_ASSERT(weight == -expectedWeight, "Bad middle weight");
    }
  else
    {
    DAX_TEST_ASSERT(weight == expectedWeight, "Bad middle weight");
    }
}

static void TestDerivativeWeightsVoxel()
{
  std::cout << "In TestDerivativeWeightsVoxel" << std::endl;
  typedef dax::exec::CellVoxel CellType;

  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < CellType::NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
        dax::exec::internal::derivativeWeightsVoxel(pcoords);

    for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
      {
      dax::Vector3 vertexPCoords =
          dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

      TestWeightOnVertex(weights[weightIndex],
                         pcoords,
                         vertexPCoords,
                         CellType::TOPOLOGICAL_DIMENSIONS);
      }
    }

  // Check for Derivative at middle.
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
      dax::exec::internal::derivativeWeightsVoxel(
        dax::exec::ParametricCoordinates<CellType>::Center());
  for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
    {
    dax::Vector3 vertexPCoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

    TestWeightInMiddle(weights[weightIndex][0],
                       vertexPCoords[0],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    TestWeightInMiddle(weights[weightIndex][1],
                       vertexPCoords[1],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    TestWeightInMiddle(weights[weightIndex][2],
                       vertexPCoords[2],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    }
}

static void TestDerivativeWeightsHexahedron()
{
  std::cout << "In TestDerivativeWeightsHexahedron" << std::endl;
  typedef dax::exec::CellHexahedron CellType;

  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < CellType::NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
        dax::exec::internal::derivativeWeightsHexahedron(pcoords);

    for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
      {
      dax::Vector3 vertexPCoords =
          dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

      TestWeightOnVertex(weights[weightIndex],
                         pcoords,
                         vertexPCoords,
                         CellType::TOPOLOGICAL_DIMENSIONS);
      }
    }

  // Check for Derivative at middle.
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
      dax::exec::internal::derivativeWeightsHexahedron(
        dax::exec::ParametricCoordinates<CellType>::Center());
  for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
    {
    dax::Vector3 vertexPCoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

    TestWeightInMiddle(weights[weightIndex][0],
                       vertexPCoords[0],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    TestWeightInMiddle(weights[weightIndex][1],
                       vertexPCoords[1],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    TestWeightInMiddle(weights[weightIndex][2],
                       vertexPCoords[2],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    }
}

static void TestDerivativeWeightsQuadrilateral()
{
  std::cout << "In TestDerivativeWeightsQuadrilateral" << std::endl;
  typedef dax::exec::CellQuadrilateral CellType;

  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0; vertexIndex < CellType::NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
        dax::exec::internal::derivativeWeightsQuadrilateral(pcoords);

    for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
      {
      dax::Vector3 vertexPCoords =
          dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

      TestWeightOnVertex(weights[weightIndex],
                         pcoords,
                         vertexPCoords,
                         CellType::TOPOLOGICAL_DIMENSIONS);
      }
    }

  // Check for Derivative at middle.
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
      dax::exec::internal::derivativeWeightsQuadrilateral(
        dax::exec::ParametricCoordinates<CellType>::Center());
  for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
    {
    dax::Vector3 vertexPCoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

    TestWeightInMiddle(weights[weightIndex][0],
                       vertexPCoords[0],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    TestWeightInMiddle(weights[weightIndex][1],
                       vertexPCoords[1],
                       CellType::TOPOLOGICAL_DIMENSIONS);
    }
}

void TestDerivativeWeights()
{
  TestDerivativeWeightsVoxel();
  TestDerivativeWeightsHexahedron();
  TestDerivativeWeightsQuadrilateral();
}

} // Anonymous namespace

int UnitTestDerivativeWeights(int, char *[])
{
  return dax::internal::Testing::Run(TestDerivativeWeights);
}
