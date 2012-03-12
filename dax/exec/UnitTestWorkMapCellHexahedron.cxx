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

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

extern void TestCellHexahedron(const dax::exec::CellHexahedron cell,
                               const dax::exec::CellVoxel Hexahedron);

extern dax::internal::TopologyUnstructured<dax::exec::CellHexahedron> make_ugrid(
    const dax::internal::TopologyUniform& uniform,
    std::vector<dax::Vector3>& points,
    std::vector<dax::Id>& topology);

namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler ErrorHandler
  = dax::exec::internal::ErrorHandler(dax::internal::DataArray<char>());

}  // Anonymous namespace


static void TestMapCellHexahedron(
  dax::exec::WorkMapCell<dax::exec::CellHexahedron> &work,
  const dax::internal::TopologyUniform &gridstruct,
  dax::Id cellFlatIndex)
{
  DAX_TEST_ASSERT(work.GetCellIndex() == cellFlatIndex,
                  "Work object returned wrong cell index.");

  dax::Id3 dim = dax::extentDimensions(gridstruct.Extent);
  dax::Id numCells = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numCells);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[cellFlatIndex] = cellFlatIndex;

  dax::internal::DataArray<dax::Scalar> fieldArray(&fieldData.at(0),
                                                   fieldData.size());
  dax::exec::FieldCell<dax::Scalar> field(fieldArray);

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

  dax::internal::TopologyUniform gridstruct;

  std::vector<dax::Id> topo;
  std::vector<dax::Vector3> points;
  dax::internal::TopologyUnstructured<dax::exec::CellHexahedron> ugrid;

  {
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  ugrid = make_ugrid(gridstruct,points,topo);

  dax::exec::WorkMapCell<dax::exec::CellHexahedron> work(ugrid,ErrorHandler);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    work.SetCellIndex(flatIndex);
    TestMapCellHexahedron(work, gridstruct, flatIndex);
    }
  }

  {
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  ugrid = make_ugrid(gridstruct,points,topo);

  dax::exec::WorkMapCell<dax::exec::CellHexahedron> work(ugrid,ErrorHandler);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    work.SetCellIndex(flatIndex);
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
