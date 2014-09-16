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

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/MarchingCubes.h>

#include <iostream>

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/cont/testing/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id ISOVALUE = 70;

template<typename PointCoordinatesArrayType,
         typename FieldArrayType>
DAX_CONT_EXPORT
void CheckFieldInterpolation(PointCoordinatesArrayType pointCoordinatesArray,
                             FieldArrayType fieldArray,
                             dax::Vector3 gradient)
{
  typedef typename PointCoordinatesArrayType::PortalConstControl
      PointCoordinatesPortalType;
  typedef typename FieldArrayType::PortalConstControl FieldPortalType;

  PointCoordinatesPortalType pointCoordinatesPortal =
      pointCoordinatesArray.GetPortalConstControl();
  FieldPortalType fieldPortal = fieldArray.GetPortalConstControl();

  DAX_TEST_ASSERT(pointCoordinatesPortal.GetNumberOfValues() ==
                  fieldPortal.GetNumberOfValues(),
                  "Field has wrong number of values.");
  dax::Id numPoints = pointCoordinatesPortal.GetNumberOfValues();

  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    dax::Vector3 coords = pointCoordinatesPortal.Get(pointIndex);
    dax::Scalar expectedScalar = dax::dot(coords, gradient);
    dax::Scalar actualScalar = fieldPortal.Get(pointIndex);
    DAX_TEST_ASSERT(test_equal(expectedScalar, actualScalar),
                    "Got bad scalar value.");
    }
}

//-----------------------------------------------------------------------------
struct TestMarchingCubesWorklet
{
  typedef dax::cont::ArrayContainerControlTagBasic ArrayContainer;
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

  typedef dax::CellTagTriangle CellType;

  typedef dax::cont::UnstructuredGrid<
      CellType,ArrayContainer,ArrayContainer,DeviceAdapter>
      UnstructuredGridType;

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  template<class InputGridType>
  DAX_CONT_EXPORT
  void operator()(const InputGridType&) const
    {
    dax::cont::testing::TestGrid<InputGridType,ArrayContainer,DeviceAdapter>
        inGrid(DIM);
    UnstructuredGridType outGrid;

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    dax::Id numPoints = inGrid->GetNumberOfPoints();
    std::vector<dax::Scalar> field(numPoints);
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.GetPointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter>
        fieldHandle = dax::cont::make_ArrayHandle(field,
                                                  ArrayContainer(),
                                                  DeviceAdapter());

    // Make a secondary field to make sure we can interpolate this field on
    // the contour in addition to the point coordinates.
    dax::Vector3 secondaryGradient = dax::make_Vector3(-0.5, 1.0, 2.0);
    std::vector<dax::Scalar> secondaryField(numPoints);
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.GetPointCoordinates(pointIndex);
      secondaryField[pointIndex] = dax::dot(coordinates, secondaryGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter>
        secondaryFieldHandle = dax::cont::make_ArrayHandle(secondaryField,
                                                           ArrayContainer(),
                                                           DeviceAdapter());
    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter>
        interpolatedSecondaryField;

    const dax::Scalar isoValue = ISOVALUE;

    try
      {
      typedef dax::cont::ArrayHandle<dax::Id, ArrayContainer, DeviceAdapter>
                    CountHandleType;

      //construct the two dispatcher that will be used to do the marching cubes
      typedef  dax::cont::DispatcherMapCell<
                          dax::worklet::MarchingCubesCount > CellDispatcher;
      typedef  dax::cont::DispatcherGenerateInterpolatedCells<
                  dax::worklet::MarchingCubesGenerate > InterpolatedDispatcher;

      //run the first step
      std::cout << "Count how many triangles are to be generated." << std::endl;
      CountHandleType count; //array handle for the first step count
      CellDispatcher cellDispatcher( (dax::worklet::MarchingCubesCount(isoValue)) );
      cellDispatcher.Invoke( inGrid.GetRealGrid(),
                             fieldHandle,
                             count);

      std::cout << "Generate the contour triangles with duplicate points."
                << std::endl;
      //construct the topology generation worklet
      InterpolatedDispatcher interpDispatcher( count,
                              dax::worklet::MarchingCubesGenerate(isoValue) );

      interpDispatcher.SetRemoveDuplicatePoints(false);
      //so we can use the count again
      interpDispatcher.SetReleaseCount(false);

      //run the second step
      interpDispatcher.Invoke(inGrid.GetRealGrid(),
                              outGrid,
                              fieldHandle);

      const dax::Id valid_num_points =
            dax::CellTraits<CellType>::NUM_VERTICES * outGrid.GetNumberOfCells();
      DAX_TEST_ASSERT(valid_num_points==outGrid.GetNumberOfPoints(),
                      "Incorrect number of points in the output grid when not merging");

      std::cout << "Check interpolation of secondary field." << std::endl;
      interpDispatcher.CompactPointField(secondaryFieldHandle,
                                         interpolatedSecondaryField);
      CheckFieldInterpolation(outGrid.GetPointCoordinates(),
                              interpolatedSecondaryField,
                              secondaryGradient);

      std::cout
          << "Generate the contour again, this time removing duplicate points."
          << std::endl;
      interpDispatcher.SetRemoveDuplicatePoints(true);
      //run the second step again with point merging

      UnstructuredGridType secondOutGrid;
      interpDispatcher.Invoke(inGrid.GetRealGrid(),
                              secondOutGrid,
                              fieldHandle);

      std::cout << "Number of points is " << secondOutGrid.GetNumberOfPoints()
                << std::endl;
      std::cout << "valid_num_points = " << valid_num_points << std::endl;

      //45 is the number I got from running the algorithm by hand and seeing
      //the number of unique points
      const dax::Id NumberOfUniquePoints = 45;
      DAX_TEST_ASSERT(NumberOfUniquePoints == secondOutGrid.GetNumberOfPoints() &&
                      NumberOfUniquePoints != valid_num_points,
          "We didn't merge to the correct number of points");

      std::cout << "Check interpolation of secondary field." << std::endl;
      interpDispatcher.CompactPointField(secondaryFieldHandle,
                                         interpolatedSecondaryField);
      CheckFieldInterpolation(secondOutGrid.GetPointCoordinates(),
                              interpolatedSecondaryField,
                              secondaryGradient);
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
    }
};


//-----------------------------------------------------------------------------
void TestMarchingCubes()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(
        TestMarchingCubesWorklet(),
        dax::testing::Testing::CellCheckHexahedron());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMarchingCubes(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestMarchingCubes);
}
