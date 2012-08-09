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

#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>

namespace {

template<class CellType>
void TestWeightOnVertex(dax::Vector3 weight,
                        dax::Vector3 derivativePCoord,
                        dax::Vector3 vertexPCoord)
{
  dax::Vector3 signs = 2.0*vertexPCoord - dax::make_Vector3(1.0, 1.0, 1.0);
  if (CellType::TOPOLOGICAL_DIMENSIONS < 3) { signs[2] = 0.0; }

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

// Wedge is linear in two dimensions and nonlinear in third.  Has
// weird derivatives.
template<>
void TestWeightOnVertex<dax::exec::CellWedge>(dax::Vector3 weight,
                                              dax::Vector3 derivativePCoord,
                                              dax::Vector3 vertexPCoord)
{
  dax::Scalar zFactor = ((vertexPCoord[2] == 0.0)
                         ? (1.0 - derivativePCoord[2]): derivativePCoord[2]);
  dax::Scalar zSign = ((vertexPCoord[2] == 0.0) ? -1.0 : 1.0);
  if ((vertexPCoord[0] == 0.0) && (vertexPCoord[1] == 0.0))
    {
    DAX_TEST_ASSERT(weight[0] == -zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == -zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(
          weight[2] == (1 - derivativePCoord[0] - derivativePCoord[1]) * zSign,
          "Bad vertex weight.");
    }
  else if ((vertexPCoord[0] == 0.0) && (vertexPCoord[1] == 1.0))
    {
    DAX_TEST_ASSERT(weight[0] == 0.0, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[2] == derivativePCoord[1]*zSign,
                    "Bad vertex weight.");
    }
  else if ((vertexPCoord[0] == 1.0) && (vertexPCoord[1] == 0.0))
    {
    DAX_TEST_ASSERT(weight[0] == zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == 0.0, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[2] == derivativePCoord[0]*zSign,
                    "Bad vertex weight.");
    }
  else
    {
    DAX_TEST_FAIL(
          "Got parametric coordinates that do not seem to be on a vertex.");
    }
}

template<class CellType>
void TestWeightInMiddle(dax::Vector3 weight,
                        dax::Vector3 vertexPCoord)
{
  dax::Scalar expectedWeight = 0.0;
  switch (CellType::TOPOLOGICAL_DIMENSIONS)
    {
    case 3:  expectedWeight = 0.25;  break;
    case 2:  expectedWeight = 0.5;   break;
    default: DAX_TEST_FAIL("Unknown dimensions.");
    }

  for (int component = 0;
       component < CellType::TOPOLOGICAL_DIMENSIONS;
       component++)
    {
    if (vertexPCoord[component] < 0.5)
      {
      DAX_TEST_ASSERT(weight[component] == -expectedWeight,"Bad middle weight");
      }
    else
      {
      DAX_TEST_ASSERT(weight[component] == expectedWeight, "Bad middle weight");
      }
    }
}

template<>
void TestWeightInMiddle<dax::exec::CellWedge>(dax::Vector3 weight,
                                              dax::Vector3 vertexPCoord)
{
  dax::Scalar zFactor = 0.5;
  dax::Scalar zSign = ((vertexPCoord[2] == 0.0) ? -1.0 : 1.0);
  if ((vertexPCoord[0] == 0.0) && (vertexPCoord[1] == 0.0))
    {
    DAX_TEST_ASSERT(weight[0] == -zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == -zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(test_equal(weight[2], zSign/3), "Bad vertex weight.");
    }
  else if ((vertexPCoord[0] == 0.0) && (vertexPCoord[1] == 1.0))
    {
    DAX_TEST_ASSERT(weight[0] == 0.0, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(test_equal(weight[2], zSign/3), "Bad vertex weight.");
    }
  else if ((vertexPCoord[0] == 1.0) && (vertexPCoord[1] == 0.0))
    {
    DAX_TEST_ASSERT(weight[0] == zFactor, "Bad vertex weight.");
    DAX_TEST_ASSERT(weight[1] == 0.0, "Bad vertex weight.");
    DAX_TEST_ASSERT(test_equal(weight[2], zSign/3), "Bad vertex weight.");
    }
  else
    {
    DAX_TEST_FAIL(
          "Got parametric coordinates that do not seem to be on a vertex.");
    }
}

template<class CellType>
void TestDerivativeWeights()
{
  // Check Derivative at each corner.
  for (dax::Id vertexIndex = 0;
       vertexIndex < CellType::NUM_POINTS;
       vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
        dax::exec::internal::DerivativeWeights<CellType>(pcoords);

    for (dax::Id weightIndex = 0;
         weightIndex < CellType::NUM_POINTS;
         weightIndex++)
      {
      dax::Vector3 vertexPCoords =
          dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

      TestWeightOnVertex<CellType>(weights[weightIndex],
                                   pcoords,
                                   vertexPCoords);
      }
    }

  // Check for Derivative at middle.
  dax::Tuple<dax::Vector3,CellType::NUM_POINTS> weights =
      dax::exec::internal::DerivativeWeights<CellType>(
        dax::exec::ParametricCoordinates<CellType>::Center());
  for (dax::Id weightIndex = 0; weightIndex < CellType::NUM_POINTS; weightIndex++)
    {
    dax::Vector3 vertexPCoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[weightIndex];

    TestWeightInMiddle<CellType>(weights[weightIndex],
                                 vertexPCoords);
    }
}

// Special cases for totally linear cells with a different type of weighting.
// They don't have implementations for DerivativeWeights, and the checks in
// this test would be wrong if they did.
template<>
void TestDerivativeWeights<dax::exec::CellTetrahedron>()
{
  std::cout << "  No derivative weights for tetrahedra.  Skiping." << std::endl;
}
template<>
void TestDerivativeWeights<dax::exec::CellTriangle>()
{
  std::cout << "  No derivative weights for triangles.  Skiping." << std::endl;
}
template<>
void TestDerivativeWeights<dax::exec::CellLine>()
{
  std::cout << "  No derivative weights for lines.  Skiping." << std::endl;
}
template<>
void TestDerivativeWeights<dax::exec::CellVertex>()
{
  std::cout << "  No derivative weights for vertices.  Skiping." << std::endl;
}

struct TestDerivativeWeightsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &) const {
    TestDerivativeWeights<typename TopologyGenType::CellType>();
  }
};

void TestAllDerivativeWeights()
{
  dax::exec::internal::TryAllTopologyTypes(TestDerivativeWeightsFunctor());
}

} // Anonymous namespace

int UnitTestDerivativeWeights(int, char *[])
{
  return dax::internal::Testing::Run(TestAllDerivativeWeights);
}
