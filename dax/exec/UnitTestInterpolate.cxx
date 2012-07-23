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
#include <dax/exec/Interpolate.h>

#include <dax/exec/ParametricCoordinates.h>
#include <dax/exec/VectorOperations.h>
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

struct Add
{
  template<typename T>
  T operator()(const T &a, const T &b) const {
    return a + b;
  }
};

template<class CellType>
void TestInterpolateSpecial(
    const CellType &cell,
    const dax::Tuple<dax::Scalar,CellType::NUM_POINTS> &pointFieldValues)
{
  const dax::Id NUM_POINTS = CellType::NUM_POINTS;

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];
    dax::Scalar interpolatedValue =
        dax::exec::CellInterpolate(cell, pointFieldValues, pcoords);
    dax::Scalar expectedValue = pointFieldValues[vertexIndex];
    DAX_TEST_ASSERT(test_equal(interpolatedValue, expectedValue),
                    "Interpolation wrong on vertex.");
    }

  dax::Scalar interpolatedValue =
      dax::exec::CellInterpolate(
        cell,
        pointFieldValues,
        dax::exec::ParametricCoordinates<CellType>::Center());
  dax::Scalar expectedValue =
      dax::exec::VectorReduce(pointFieldValues, Add())/NUM_POINTS;
  DAX_TEST_ASSERT(test_equal(interpolatedValue, expectedValue),
                  "Interpolation wrong at center.");
}

template<class CellType>
void TestInterpolateSample(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoordinates,
    const dax::Tuple<dax::Scalar,CellType::NUM_POINTS> &pointFieldValues,
    const LinearField &fieldValues)
{
  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0; pcoords[2] <= 1.0; pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0; pcoords[1] <= 1.0; pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0; pcoords[0] <= 1.0; pcoords[0] += 0.25)
        {
        dax::Scalar interpolatedValue
            = dax::exec::CellInterpolate(cell, pointFieldValues, pcoords);

        dax::Vector3 wcoords
            = dax::exec::CellInterpolate(cell, vertexCoordinates, pcoords);

        dax::Scalar trueValue = fieldValues.GetValue(wcoords);

        DAX_TEST_ASSERT(test_equal(interpolatedValue, trueValue),
                        "Bad interpolated value");
        }
      }
    }
}

template<class CellType>
void TestInterpolateCell(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoordinates,
    const dax::Tuple<dax::Scalar,CellType::NUM_POINTS> &pointFieldValues,
    const LinearField &fieldValues)
{
  TestInterpolateSpecial(cell, pointFieldValues);
  TestInterpolateSample(cell, vertexCoordinates, pointFieldValues, fieldValues);
}

template<class TopologyGenType>
void TestInterpolate(const TopologyGenType &topology,
                     const LinearField &fieldValues)
{
  typedef typename TopologyGenType::CellType CellType;

  std::vector<dax::Scalar> fieldArray;
  CreatePointField(topology, fieldValues, fieldArray);

  dax::Id numCells = topology.GetNumberOfCells();
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    TestInterpolateCell(topology.GetCell(cellIndex),
                        topology.GetCellVertexCoordinates(cellIndex),
                        topology.GetFieldValuesIterator(cellIndex,
                                                        fieldArray.begin()),
                        fieldValues);
    }
}

struct TestInterpolateFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) {
    LinearField fieldValues;

    std::cout << "Very simple field." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(1.0, 1.0, 1.0);
    fieldValues.OriginValue = 0.0;
    TestInterpolate(topology, fieldValues);

    std::cout << "Uneven gradient." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(0.25, 14.0, 11.125);
    fieldValues.OriginValue = -7.0;
    TestInterpolate(topology, fieldValues);

    std::cout << "Negative gradient directions." << std::endl;
    fieldValues.Gradient = dax::make_Vector3(-11.125, -0.25, 14.0);
    fieldValues.OriginValue = 5.0;
    TestInterpolate(topology, fieldValues);
  }
};

void TestAllInterpolate()
{
  dax::exec::internal::TryAllTopologyTypes(TestInterpolateFunctor());
}

} // Anonymous namespace

int UnitTestInterpolate(int, char *[])
{
  return dax::internal::Testing::Run(TestAllInterpolate);
}
