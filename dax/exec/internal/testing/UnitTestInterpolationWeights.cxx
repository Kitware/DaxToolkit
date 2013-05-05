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

#include <dax/testing/Testing.h>

namespace {

template<class CellTag>
void CheckVertexWeights(CellTag)
{
  std::cout << "  Checking weights of vertices." << std::endl;

  const dax::Id NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;
  dax::Tuple<dax::Vector3,NUM_VERTICES> vertexParametricCoords =
      dax::exec::ParametricCoordinates<CellTag>::Vertex();

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
    {
    dax::Vector3 pcoords = vertexParametricCoords[vertexIndex];

    dax::Tuple<dax::Scalar, NUM_VERTICES> weights =
        dax::exec::internal::InterpolationWeights(pcoords, CellTag());

    for (dax::Id weightIndex = 0; weightIndex < NUM_VERTICES; weightIndex++)
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

template<class CellTag>
void CheckCenterWeight(CellTag)
{
  std::cout << "  Checking weight at center." << std::endl;

  const dax::Id NUM_VERTICES = dax::CellTraits<CellTag>::NUM_VERTICES;

  dax::Tuple<dax::Scalar, NUM_VERTICES> weights =
      dax::exec::internal::InterpolationWeights(
        dax::exec::ParametricCoordinates<CellTag>::Center(), CellTag());

  for (dax::Id weightIndex = 0; weightIndex < NUM_VERTICES; weightIndex++)
    {
    DAX_TEST_ASSERT(test_equal(weights[weightIndex],
                               dax::Scalar(1.0/NUM_VERTICES)),
                    "Got bad interpolation weight");
    }
}

//-----------------------------------------------------------------------------
template<class CellTag>
void TestInterpolationWeights(CellTag)
{
  CheckVertexWeights(CellTag());

  CheckCenterWeight(CellTag());
}

//-----------------------------------------------------------------------------
struct TestInterpolationWeightsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &) const {
    TestInterpolationWeights(typename TopologyGenType::CellTag());
  }
};

void TestAllInterpolationWeights()
{
  dax::exec::internal::TryAllTopologyTypes(TestInterpolationWeightsFunctor());
}

} // anonymous namespace

int UnitTestInterpolationWeights(int, char *[])
{
  return dax::testing::Testing::Run(TestAllInterpolationWeights);
}
