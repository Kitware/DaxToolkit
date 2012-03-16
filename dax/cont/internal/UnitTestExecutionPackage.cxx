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

#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

namespace
{
const dax::Id ARRAY_SIZE = 10;

//TODO: Make tests for ExecutionPackageField* classes.

void TestExecutionPackageGrid()
{
  std::cout << "Test package UniformGrid." << std::endl;
  dax::cont::UniformGrid uniformGrid;
  dax::Id3 minExtent = dax::make_Id3(-ARRAY_SIZE, -ARRAY_SIZE, -ARRAY_SIZE);
  dax::Id3 maxExtent = dax::make_Id3(ARRAY_SIZE, ARRAY_SIZE, ARRAY_SIZE);
  uniformGrid.SetExtent(minExtent, maxExtent);
  dax::cont::internal::ExecutionPackageGrid<dax::cont::UniformGrid>
      uniformGridPackage(uniformGrid);

  dax::internal::TopologyUniform uniformStructure
      = uniformGridPackage.GetExecutionObject();
  DAX_TEST_ASSERT(uniformStructure.Extent.Min == minExtent,
              "Bad uniform grid structure");
  DAX_TEST_ASSERT(uniformStructure.Extent.Max == maxExtent,
              "Bad uniform grid structure");
}

void TestExecutionPackages()
{
  TestExecutionPackageGrid();
}

} // Anonymous namespace

int UnitTestExecutionPackage(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestExecutionPackages);
}
