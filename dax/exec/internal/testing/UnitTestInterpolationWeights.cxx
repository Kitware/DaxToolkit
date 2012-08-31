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

#include <dax/exec/ParametricCoordinates.h>

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/internal/testing/Testing.h>

namespace {

template<class CellType>
void CheckVertexWeights(
    )
{
  std::cout << "  Checking weights of vertices." << std::endl;

  const dax::Id NUM_POINTS = CellType::NUM_POINTS;
  dax::Tuple<dax::Vector3,NUM_POINTS> vertexParametricCoords =
      dax::exec::ParametricCoordinates<CellType>::Vertex();

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords = vertexParametricCoords[vertexIndex];

    dax::Tuple<dax::Scalar, NUM_POINTS> weights =
        dax::exec::internal::InterpolationWeights<CellType>(pcoords);

    for (dax::Id weightIndex = 0; weightIndex < NUM_POINTS; weightIndex++)
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
}

template<class CellType>
void CheckCenterWeight()
{
  std::cout << "  Checking weight at center." << std::endl;

  const dax::Id NUM_POINTS = CellType::NUM_POINTS;

  dax::Tuple<dax::Scalar, NUM_POINTS> weights =
      dax::exec::internal::InterpolationWeights<CellType>(
        dax::exec::ParametricCoordinates<CellType>::Center());

  for (dax::Id weightIndex = 0; weightIndex < NUM_POINTS; weightIndex++)
    {
    DAX_TEST_ASSERT(test_equal(weights[weightIndex],
                               dax::Scalar(1.0/NUM_POINTS)),
                    "Got bad interpolation weight");
    }
}

//-----------------------------------------------------------------------------
template<class CellType>
void TestInterpolationWeights()
{
  CheckVertexWeights<CellType>();

  CheckCenterWeight<CellType>();
}

//-----------------------------------------------------------------------------
struct TestInterpolationWeightsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &) const {
    TestInterpolationWeights<typename TopologyGenType::CellType>();
  }
};

void TestAllInterpolationWeights()
{
  dax::exec::internal::TryAllTopologyTypes(TestInterpolationWeightsFunctor());
}

} // anonymous namespace

int UnitTestInterpolationWeights(int, char *[])
{
  return dax::internal::Testing::Run(TestAllInterpolationWeights);
}
