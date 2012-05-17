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

// These two includes ensure that we can read the execution data from
// here in the control environment.
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/WorkEmpty.h>

#include <dax/cont/internal/TestingGridGenerator.h>
#include <dax/cont/internal/Testing.h>

namespace
{
const dax::Id ARRAY_SIZE = 10;

typedef dax::cont::DeviceAdapterSerial
    ::ExecutionAdapter<dax::cont::ArrayContainerControlBasic>
      ExecutionAdapter;

template<class IteratorType1, class IteratorType2>
void CompareArrays(IteratorType1 begin1,
                   IteratorType1 end1,
                   IteratorType2 begin2)
{
  IteratorType1 iter1 = begin1;
  IteratorType2 iter2 = begin2;
  for (; iter1 != end1; iter1++, iter2++)
    {
    DAX_TEST_ASSERT(test_equal(*iter1, *iter2), "Array value not the same.");
    }
}

template<class GridType, class TopologyType>
void BasicGridComparison(const GridType &grid, const TopologyType &topology)
{
  std::cout << "Basic metadata comparison." << std::endl;

  DAX_TEST_ASSERT(grid.GetNumberOfPoints()
                  == dax::exec::internal::numberOfPoints(topology),
                  "Number of points wrong.");
  DAX_TEST_ASSERT(grid.GetNumberOfCells()
                  == dax::exec::internal::numberOfCells(topology),
                  "Number of cells wrong.");
}

void TestExecutionGrid(const dax::cont::UniformGrid &grid)
{
  std::cout << "****Test package UniformGrid" << std::endl;

  dax::exec::internal::TopologyUniform topology
      = dax::cont::internal::ExecutionPackageGrid::GetExecutionObject(grid);

  BasicGridComparison(grid, topology);

  std::cout << "Structure comparison." << std::endl;
  DAX_TEST_ASSERT(test_equal(grid.GetOrigin(), topology.Origin),
                  "Bad origin.");
  DAX_TEST_ASSERT(test_equal(grid.GetSpacing(), topology.Spacing),
                  "Bad Spacing.");
  DAX_TEST_ASSERT(grid.GetExtent().Min == topology.Extent.Min,
                  "Bad extent.");
  DAX_TEST_ASSERT(grid.GetExtent().Max == topology.Extent.Max,
                  "Bad extent.");

  std::cout << "Done testing UniformGrid" << std::endl;
}

void TestExecutionGrid(
    const dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> &grid)
{
  std::cout << "****Test package UnstructuredGrid" << std::endl;

  typedef dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> GridType;

  dax::exec::internal
      ::TopologyUnstructured<GridType::CellType, GridType::ExecutionAdapter>
      topology
      = dax::cont::internal::ExecutionPackageGrid::GetExecutionObject(grid);

  BasicGridComparison(grid, topology);

  std::cout << "Check point coordinates." << std::endl;
  CompareArrays(grid.GetPointCoordinates().GetIteratorConstControlBegin(),
                grid.GetPointCoordinates().GetIteratorConstControlEnd(),
                topology.PointCoordinates);

  std::cout << "Check cell connections." << std::endl;
  CompareArrays(grid.GetCellConnections().GetIteratorConstControlBegin(),
                grid.GetCellConnections().GetIteratorConstControlEnd(),
                topology.CellConnections);

  std::cout << "Done testing UnstructuredGrid" << std::endl;
}

template<class GridType>
void TestExecutionGrid()
{
  dax::cont::internal::TestGrid<GridType> generator(ARRAY_SIZE);
  TestExecutionGrid(generator.GetRealGrid());
}

void TestExecutionPackageGrid()
{
  TestExecutionGrid<dax::cont::UniformGrid>();
  TestExecutionGrid<dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> >();
}

//-----------------------------------------------------------------------------

template<class FieldType>
void TestExecutionField(dax::exec::internal::FieldAccessInputTag)
{
  std::cout << "****Checking input field." << std::endl;

  typedef typename FieldType::ValueType ValueType;

  // Create input.
  std::vector<ValueType> originalData(ARRAY_SIZE);
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    originalData[i] = ValueType(10*i + 0.01*(i+1));
    }
  dax::cont::ArrayHandle<ValueType> inputArray(
        &originalData.front(), (&originalData.back()) + 1);

  FieldType field = dax::cont::internal::ExecutionPackageField
      ::GetExecutionObject<FieldType>(inputArray, ARRAY_SIZE);

  dax::exec::internal::WorkEmpty<ExecutionAdapter> dummywork =
      dax::exec::internal::WorkEmpty<ExecutionAdapter>(
        typename ExecutionAdapter::ErrorHandler());

  for(dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    ValueType orig = originalData[i];
    ValueType f
        = dax::exec::internal::FieldAccess::GetField(field, i, dummywork);
    DAX_TEST_ASSERT(orig == f, "Array did not get pushed to field.");
    }
}

template<class FieldType>
void TestExecutionField(dax::exec::internal::FieldAccessOutputTag)
{
  std::cout << "****Checking output field." << std::endl;

  typedef typename FieldType::ValueType ValueType;

  // Create field.
  dax::cont::ArrayHandle<ValueType> array;

  FieldType field = dax::cont::internal::ExecutionPackageField
      ::GetExecutionObject<FieldType>(array, ARRAY_SIZE);

  std::cout << "Checking that field create allocated array." << std::endl;
  DAX_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE,
                  "Array not allocated as expected.");

  std::cout << "Filling array." << std::endl;
  dax::exec::internal::WorkEmpty<ExecutionAdapter> dummywork =
      dax::exec::internal::WorkEmpty<ExecutionAdapter>(
        typename ExecutionAdapter::ErrorHandler());

  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    dax::exec::internal::FieldAccess::SetField(field,i,ValueType(i),dummywork);
    }

  std::cout << "Checking that field comes back to control." << std::endl;
  typedef typename dax::cont::ArrayHandle<ValueType>::IteratorConstControl
      IteratorType;
  dax::Id i = 0;
  for (IteratorType iter = array.GetIteratorConstControlBegin();
       iter != array.GetIteratorConstControlEnd();
       iter++)
    {
    DAX_TEST_ASSERT(*iter == ValueType(i),
                    "Did not get back right values in control.");
    i++;
    }
}

template<class FieldType>
void TestExecutionField()
{
  TestExecutionField<FieldType>(typename FieldType::AccessTag());
}

template<template <typename, class> class FieldType>
void TestExecutionField()
{
  TestExecutionField<FieldType<dax::Scalar, ExecutionAdapter> >();
}

template<template <class> class FieldType>
void TestExecutionField()
{
  TestExecutionField<FieldType<ExecutionAdapter> >();
}

void TestExecutionPackageField()
{
  TestExecutionField<dax::exec::FieldIn>();
  TestExecutionField<dax::exec::FieldOut>();
  TestExecutionField<dax::exec::FieldPointIn>();
  TestExecutionField<dax::exec::FieldPointOut>();
  TestExecutionField<dax::exec::FieldCellIn>();
  TestExecutionField<dax::exec::FieldCellOut>();
  TestExecutionField<dax::exec::FieldCoordinatesIn>();
  TestExecutionField<dax::exec::FieldCoordinatesOut>();
}

//-----------------------------------------------------------------------------
void TestExecutionPackages()
{
  TestExecutionPackageGrid();
  TestExecutionPackageField();
}

} // Anonymous namespace

int UnitTestExecutionPackage(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestExecutionPackages);
}
