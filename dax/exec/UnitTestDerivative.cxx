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

#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
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
dax::exec::FieldPointIn<dax::Scalar, typename TopologyGenType::ExecutionAdapter>
CreatePointField(const TopologyGenType &topology,
                 const LinearField &fieldValues,
                 std::vector<dax::Scalar> &fieldArray)
{
  typedef typename TopologyGenType::CellType CellType;
  typedef typename TopologyGenType::ExecutionAdapter ExecutionAdapter;

  dax::exec::FieldCoordinatesIn<ExecutionAdapter> coordField
      = topology.GetCoordinates();

  dax::exec::FieldPointOut<dax::Scalar, ExecutionAdapter> field
      = dax::exec::internal::CreateField<dax::exec::FieldPointOut>(topology,
                                                                   fieldArray);

  dax::Id numPoints = topology.GetNumberOfPoints();
  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    dax::exec::WorkMapField<CellType, ExecutionAdapter> work =
        dax::exec::internal::CreateWorkMapField(topology, pointIndex);
    dax::Vector3 coordinates = work.GetFieldValue(coordField);
    dax::Scalar value = fieldValues.GetValue(coordinates);
    work.SetFieldValue(field, value);
    }

  return dax::exec::internal::CreateField<dax::exec::FieldPointIn>(topology,
                                                                   fieldArray);
}

template<class CellType>
void TestGradientResult(
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS>
        &daxNotUsed(pointCoordinates),
    const dax::Vector3 computedDerivative,
    const LinearField &fieldValues)
{
  DAX_TEST_ASSERT(test_equal(computedDerivative, fieldValues.Gradient),
                  "Bad derivative");
}

template<>
void TestGradientResult<dax::exec::CellTriangle>(
    const dax::Tuple<dax::Vector3, dax::exec::CellTriangle::NUM_POINTS>
        &pointCoordinates,
    const dax::Vector3 computedDerivative,
    const LinearField &fieldValues)
{
  dax::Vector3 normal = dax::math::TriangleNormal(
        pointCoordinates[0], pointCoordinates[1], pointCoordinates[2]);
  dax::math::Normalize(normal);
  dax::Vector3 expectedGradient =
      fieldValues.Gradient - dax::dot(fieldValues.Gradient,normal)*normal;
  DAX_TEST_ASSERT(test_equal(computedDerivative, expectedGradient),
                  "Bad derivative");
}

template<class CellType, class ExecutionAdapter>
void TestDerivativeCell(
    const dax::exec::WorkMapCell<CellType, ExecutionAdapter> &work,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    const dax::exec::FieldPointIn<dax::Scalar, ExecutionAdapter> &scalarField,
    const LinearField &fieldValues)
{
  CellType cell = work.GetCell();

  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0; pcoords[2] <= 1.0; pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0; pcoords[1] <= 1.0; pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0; pcoords[0] <= 1.0; pcoords[0] += 0.25)
        {
        dax::Vector3 computedDerivative
            = dax::exec::cellDerivative(work,
                                        cell,
                                        pcoords,
                                        coordField,
                                        scalarField);
        TestGradientResult<CellType>(work.GetFieldValues(coordField),
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
  typedef typename TopologyGenType::ExecutionAdapter ExecutionAdapter;

  dax::exec::FieldCoordinatesIn<ExecutionAdapter> coordField =
      topology.GetCoordinates();

  std::vector<dax::Scalar> fieldArray;
  dax::exec::FieldPointIn<dax::Scalar,ExecutionAdapter> scalarField
      = CreatePointField(topology, fieldValues, fieldArray);

  dax::Id numCells = topology.GetNumberOfCells();
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    dax::exec::WorkMapCell<CellType,ExecutionAdapter> work =
        dax::exec::internal::CreateWorkMapCell(topology, cellIndex);
    TestDerivativeCell(work, coordField, scalarField, fieldValues);
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
