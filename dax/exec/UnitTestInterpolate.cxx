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
#include <dax/exec/WorkMapCell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/TestExecutionAdapter.h>

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

} // Anonymous namespace

const dax::Id bufferSize = 1024*1024;
static dax::Scalar fieldBuffer[bufferSize];

template<class CellType>
static dax::exec::FieldPointIn<dax::Scalar, TestExecutionAdapter>
CreatePointField(
    const typename CellType::template GridStructures<TestExecutionAdapter>::TopologyType &topology,
    const dax::exec::FieldCoordinatesIn<TestExecutionAdapter> &coordField,
    const LinearField &fieldValues,
    dax::Id numPoints)
{
  DAX_TEST_ASSERT(bufferSize >= numPoints,
                  "Internal test error.  Buffer not large enough");

  // Fill field.
  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    dax::exec::WorkMapField<CellType, TestExecutionAdapter>
        work(topology, pointIndex, TestExecutionAdapter());
    dax::Vector3 coordinates = work.GetFieldValue(coordField);
    dax::Scalar fieldValue = fieldValues.GetValue(coordinates);
    fieldBuffer[pointIndex] = fieldValue;
    }

  // Create object.
  dax::exec::FieldPointIn<dax::Scalar, TestExecutionAdapter> field(fieldBuffer);

  return field;
}

template<class CellType, class ExecutionAdapter>
static void TestInterpolateCell(
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
        dax::Scalar interpolatedValue
            = dax::exec::cellInterpolate(work, cell, pcoords, scalarField);

        dax::Vector3 wcoords
            = dax::exec::parametricCoordinatesToWorldCoordinates(work,
                                                                 cell,
                                                                 coordField,
                                                                 pcoords);
        dax::Scalar trueValue = fieldValues.GetValue(wcoords);

        DAX_TEST_ASSERT(interpolatedValue == trueValue,
                        "Bad interpolated value");
        }
      }
    }
}

static void TestInterpolateVoxel(
    const dax::exec::internal::TopologyUniform &gridstruct,
    const LinearField &fieldValues)
{
  dax::exec::FieldCoordinatesIn<TestExecutionAdapter> coordField;
  dax::Id numPoints = dax::exec::internal::numberOfPoints(gridstruct);
  dax::exec::FieldPointIn<dax::Scalar, TestExecutionAdapter> scalarField =
      CreatePointField<dax::exec::CellVoxel>(
        gridstruct, coordField, fieldValues, numPoints);

  dax::Id numCells = dax::exec::internal::numberOfCells(gridstruct);
  for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellVoxel, TestExecutionAdapter>
        workCell(gridstruct, cellIndex, TestExecutionAdapter());
    TestInterpolateCell(workCell, coordField, scalarField, fieldValues);
    }
}

static void TestInterpolateVoxel()
{
  dax::exec::internal::TopologyUniform gridstruct;
  LinearField fieldValues;

  std::cout << "Very simple field." << std::endl;
  gridstruct.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  fieldValues.Gradient = dax::make_Vector3(1.0, 1.0, 1.0);
  fieldValues.OriginValue = 0.0;
  TestInterpolateVoxel(gridstruct, fieldValues);

  std::cout << "Uneven spacing/gradient." << std::endl;
  gridstruct.Origin = dax::make_Vector3(1.0, -0.5, 13.0);
  gridstruct.Spacing = dax::make_Vector3(2.5, 6.25, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(5, -2, -7);
  gridstruct.Extent.Max = dax::make_Id3(20, 4, 10);
  fieldValues.Gradient = dax::make_Vector3(0.25, 14.0, 11.125);
  fieldValues.OriginValue = -7.0;
  TestInterpolateVoxel(gridstruct, fieldValues);

  std::cout << "Negative gradient directions." << std::endl;
  gridstruct.Origin = dax::make_Vector3(-5.0, -5.0, -5.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  fieldValues.Gradient = dax::make_Vector3(-11.125, -0.25, 14.0);
  fieldValues.OriginValue = 5.0;
  TestInterpolateVoxel(gridstruct, fieldValues);
}

static void TestInterpolate()
{
  TestInterpolateVoxel();
}

int UnitTestInterpolate(int, char *[])
{
  return dax::internal::Testing::Run(TestInterpolate);
}
