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
#include <dax/exec/WorkMapField.h>

#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler<TestExecutionAdapter> ErrorHandler
  = dax::exec::internal::ErrorHandler<TestExecutionAdapter>(NULL, NULL);

}  // Anonymous namespace

static void TestMapFieldVoxel(
  dax::exec::WorkMapField<dax::exec::CellVoxel, TestExecutionAdapter> work,
  const dax::exec::internal::TopologyUniform &gridstruct,
  dax::Id pointFlatIndex)
{
  DAX_TEST_ASSERT(work.GetIndex() == pointFlatIndex,
                  "Work object returned wrong index.");

  dax::Id3 pointIjkIndex = dax::flatIndexToIndex3(pointFlatIndex,
                                                  gridstruct.Extent);

  dax::Id3 dim = dax::extentDimensions(gridstruct.Extent);
  dax::Id numPoints = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numPoints);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[pointFlatIndex] = pointFlatIndex;

  dax::exec::FieldPointOut<dax::Scalar, TestExecutionAdapter>
      field(&fieldData.at(0));

  dax::Scalar scalarValue = work.GetFieldValue(field);
  DAX_TEST_ASSERT(scalarValue == pointFlatIndex,
                  "Did not get expected data value.");

  work.SetFieldValue(field, static_cast<dax::Scalar>(-2));
  DAX_TEST_ASSERT(fieldData[pointFlatIndex] == -2,
                  "Field value did not set as expected.");

  dax::Vector3 expectedCoords
      = dax::make_Vector3(static_cast<dax::Scalar>(pointIjkIndex[0]),
                          static_cast<dax::Scalar>(pointIjkIndex[1]),
                          static_cast<dax::Scalar>(pointIjkIndex[2]));
  expectedCoords = gridstruct.Origin + expectedCoords * gridstruct.Spacing;

  dax::exec::FieldCoordinatesIn<TestExecutionAdapter> fieldCoords;
  dax::Vector3 coords = work.GetFieldValue(fieldCoords);

  DAX_TEST_ASSERT(expectedCoords == coords,
                  "Did not get expected point coordinates.");
}

static void TestMapFieldVoxel()
{
  std::cout << "Testing WorkMapField<CellVoxel>" << std::endl;

  {
  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(9, 9, 9);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::WorkMapField<dax::exec::CellVoxel, TestExecutionAdapter>
        work(gridstruct, flatIndex, ErrorHandler);
    TestMapFieldVoxel(work, gridstruct, flatIndex);
    }
  }

  {
  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(14, 5, 12);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    dax::exec::WorkMapField<dax::exec::CellVoxel, TestExecutionAdapter>
        work(gridstruct, flatIndex, ErrorHandler);
    TestMapFieldVoxel(work, gridstruct, flatIndex);
    }
  }
}

static void TestMapField()
{
  TestMapFieldVoxel();
}

int UnitTestWorkMapField(int, char *[])
{
  return dax::internal::Testing::Run(TestMapField);
}
