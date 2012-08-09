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
#include <dax/exec/Derivative.h>

#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>

namespace
{

/// Simple structure describing a linear field.  Has a convienience class
/// for getting values.
struct LinearField {
  dax::Vector3 Gradient;
  dax::Scalar OriginValue;

  dax::Scalar GetValue(dax::Vector3 coordinates) const {
    return dax::dot(coordinates, this->Gradient) + this->OriginValue;
  }
};

template<class TopologyGenType>
void CreatePointField(const TopologyGenType &topology,
                      const LinearField &fieldValues,
                      std::vector<dax::Scalar> &fieldArray)
{
  typedef typename TopologyGenType::CellType CellType;

  dax::Id numPoints = topology.GetNumberOfPoints();
  fieldArray.resize(numPoints);
  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    dax::Vector3 coordinates = topology.GetPointCoordinates(pointIndex);
    dax::Scalar value = fieldValues.GetValue(coordinates);
    fieldArray[pointIndex] = value;
    }
}

template<class CellType>
void TestGradientResult(
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &pointCoordinates,
    const dax::Vector3 computedDerivative,
    const LinearField &fieldValues)
{
  dax::Vector3 expectedGradient;
  if (CellType::TOPOLOGICAL_DIMENSIONS == 3)
    {
    expectedGradient = fieldValues.Gradient;
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 2)
    {
    dax::Vector3 normal = dax::exec::math::TriangleNormal(
          pointCoordinates[0], pointCoordinates[1], pointCoordinates[2]);
    dax::exec::math::Normalize(normal);
    expectedGradient =
        fieldValues.Gradient - dax::dot(fieldValues.Gradient,normal)*normal;
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 1)
    {
    dax::Vector3 direction =
        dax::exec::math::Normal(pointCoordinates[1]-pointCoordinates[0]);
    expectedGradient = direction * dax::dot(direction, fieldValues.Gradient);
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 0)
    {
    expectedGradient = dax::make_Vector3(0, 0, 0);
    }
  else
    {
    DAX_TEST_FAIL("Unknown cell dimension.");
    }

  DAX_TEST_ASSERT(test_equal(computedDerivative, expectedGradient),
                  "Bad derivative");
}

template<class CellType>
void TestDerivativeCell(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> vertCoords,
    const dax::Tuple<dax::Scalar,CellType::NUM_POINTS> scalarField,
    const LinearField &fieldValues)
{
  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0; pcoords[2] <= 1.0; pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0; pcoords[1] <= 1.0; pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0; pcoords[0] <= 1.0; pcoords[0] += 0.25)
        {
        dax::Vector3 computedDerivative
            = dax::exec::CellDerivative(cell,
                                        pcoords,
                                        vertCoords,
                                        scalarField);
        TestGradientResult<CellType>(vertCoords,
                                     computedDerivative,
                                     fieldValues);
        }
      }
    }
}

template<class TopologyGenType>
void TestDerivatives(const TopologyGenType &topology,
                     const LinearField &fieldValues)
{
  typedef typename TopologyGenType::CellType CellType;

  std::vector<dax::Scalar> fieldArray;
  CreatePointField(topology, fieldValues, fieldArray);

  dax::Id numCells = topology.GetNumberOfCells();
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    TestDerivativeCell(topology.GetCell(cellIndex),
                       topology.GetCellVertexCoordinates(cellIndex),
                       topology.GetFieldValuesIterator(cellIndex,
                                                       fieldArray.begin()),
                       fieldValues);
    }
}

struct TestDerivativesFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) {
    LinearField fieldValues;

    std::cout << "Very simple field." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(1.0, 1.0, 1.0);
    fieldValues.OriginValue = 0.0;
    TestDerivatives(topology, fieldValues);

    std::cout << "Uneven gradient." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(0.25, 14.0, 11.125);
    fieldValues.OriginValue = -7.0;
    TestDerivatives(topology, fieldValues);

    std::cout << "Negative gradient directions." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(-11.125, -0.25, 14.0);
    fieldValues.OriginValue = 5.0;
    TestDerivatives(topology, fieldValues);
  }
};

static void TestAllDerivatives()
{
  dax::exec::internal::TryAllTopologyTypes(TestDerivativesFunctor());
}

} // Anonymous namespace

int UnitTestDerivative(int, char *[])
{
  return dax::internal::Testing::Run(TestAllDerivatives);
}
