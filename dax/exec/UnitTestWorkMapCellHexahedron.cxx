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
#include <dax/exec/WorkMapCell.h>

#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

extern void TestCellHexahedron(const dax::exec::CellHexahedron cell,
                               const dax::exec::CellVoxel Hexahedron);

extern dax::exec::internal::TopologyUnstructured<dax::exec::CellHexahedron,TestExecutionAdapter>
make_ugrid(const dax::exec::internal::TopologyUniform& uniform,
           std::vector<dax::Vector3>& points,
           std::vector<dax::Id>& topology);

namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler<TestExecutionAdapter> ErrorHandler
  = dax::exec::internal::ErrorHandler<TestExecutionAdapter>(NULL, NULL);

}  // Anonymous namespace


static void TestMapCellHexahedron(
  dax::exec::WorkMapCell<dax::exec::CellHexahedron, TestExecutionAdapter> &work,
  const dax::exec::internal::TopologyUniform &gridstruct,
  dax::Id cellFlatIndex)
{
  DAX_TEST_ASSERT(work.GetCellIndex() == cellFlatIndex,
                  "Work object returned wrong cell index.");

  dax::Id3 dim = dax::extentDimensions(gridstruct.Extent);
  dax::Id numCells = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numCells);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[cellFlatIndex] = cellFlatIndex;

  dax::exec::FieldCellOut<dax::Scalar, TestExecutionAdapter>
      field(&fieldData.at(0));

  dax::Scalar scalarValue = work.GetFieldValue(field);
  DAX_TEST_ASSERT(scalarValue == cellFlatIndex,
                  "Did not get expected data value.");

  work.SetFieldValue(field, static_cast<dax::Scalar>(-2));
  DAX_TEST_ASSERT(fieldData[cellFlatIndex] == -2,
                  "Field value did not set as expected.");

  dax::exec::CellVoxel vox(gridstruct,work.GetCellIndex());
  TestCellHexahedron(work.GetCell(), vox);
}

static void TestMapCellHexahedron()
{
  std::cout << "Testing WorkMapCell<CellHexahedron>" << std::endl;

  dax::exec::internal::TopologyUniform gridstruct;

  std::vector<dax::Id> topo;
  std::vector<dax::Vector3> points;
  dax::exec::internal::TopologyUnstructured<
      dax::exec::CellHexahedron, TestExecutionAdapter> ugrid;

  {
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  ugrid = make_ugrid(gridstruct,points,topo);

  for (dax::Id flatIndex = 0;
       flatIndex < dax::exec::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellHexahedron, TestExecutionAdapter>
        work(ugrid, flatIndex, ErrorHandler);
    TestMapCellHexahedron(work, gridstruct, flatIndex);
    }
  }

  {
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  ugrid = make_ugrid(gridstruct,points,topo);

  for (dax::Id flatIndex = 0;
       flatIndex < dax::exec::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellHexahedron, TestExecutionAdapter>
        work(ugrid, flatIndex, ErrorHandler);
    TestMapCellHexahedron(work, gridstruct, flatIndex);
    }
  }
}

static void TestMapCell()
{
  TestMapCellHexahedron();
}

int UnitTestWorkMapCellHexahedron(int, char *[])
{
  return dax::internal::Testing::Run(TestMapCell);
}
