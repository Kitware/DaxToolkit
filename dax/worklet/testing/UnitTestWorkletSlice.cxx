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

#include <dax/worklet/Slice.h>

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


namespace {
const dax::Id DIM = 8;
const dax::Vector3 ORIGIN(3,2,1);
const dax::Vector3 NORMAL(1.0,0.0,0.0);

//-----------------------------------------------------------------------------
struct TestSliceWorklet
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

    try
      {
      typedef dax::cont::ArrayHandle<dax::Id, ArrayContainer, DeviceAdapter>
        ClassifyHandleType;

      //construct the two worklets that will be used to do the marching cubes
      typedef  dax::cont::DispatcherMapCell<
                          dax::worklet::SliceClassify > CellDispatcher;
      typedef  dax::cont::DispatcherGenerateInterpolatedCells<
                  dax::worklet::SliceGenerate > InterpolatedDispatcher;



      //run the first step
      ClassifyHandleType classification; //array handle for the first step classification
      CellDispatcher cellDispatcher((dax::worklet::SliceClassify(ORIGIN,NORMAL)));
      cellDispatcher.Invoke( inGrid.GetRealGrid(),
                            inGrid->GetPointCoordinates(),
                            classification);

      InterpolatedDispatcher interpDispatcher(classification,
                                dax::worklet::SliceGenerate(ORIGIN,NORMAL));
      interpDispatcher.SetRemoveDuplicatePoints(false);
      //so we can use the classification again
      interpDispatcher.SetReleaseClassification(false);

      //run the second step
      interpDispatcher.Invoke(inGrid.GetRealGrid(),
                              outGrid,
                              inGrid->GetPointCoordinates());

      //the number of valid cells for the slice operation is a constant
      //when the plane is along an axis, since know that each voxel will output
      //two triangles. When we merge duplicate points we also can safely compute
      //the number of valid points, as it will equal DIM * DIM;
      const dax::Id validNumberOfCells = ( (DIM-1)*(DIM-1) )* 2;
      const dax::Id validNumberOfPointsNoMerge = validNumberOfCells *
                                dax::CellTraits<CellType>::NUM_VERTICES;
      const dax::Id validNumberOfPointsMerge = DIM * DIM;

      DAX_TEST_ASSERT(validNumberOfCells==outGrid.GetNumberOfCells(),
             "Incorrect number of points in the output grid when not merging");
      DAX_TEST_ASSERT(validNumberOfPointsNoMerge==outGrid.GetNumberOfPoints(),
             "Incorrect number of points in the output grid when not merging");


      //run the second step again with point merging
      interpDispatcher.SetRemoveDuplicatePoints(true);
      UnstructuredGridType secondOutGrid;
      interpDispatcher.Invoke(inGrid.GetRealGrid(),
                              secondOutGrid,
                              inGrid->GetPointCoordinates());

      //Once again validate the output grid
      DAX_TEST_ASSERT(validNumberOfCells==secondOutGrid.GetNumberOfCells(),
             "Incorrect number of points in the output grid when merging");
      DAX_TEST_ASSERT(validNumberOfPointsMerge ==
                                          secondOutGrid.GetNumberOfPoints(),
             "Incorrect number of points in the output grid when merging");
      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
    }
};


//-----------------------------------------------------------------------------
void TestSlice()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(
        TestSliceWorklet(),
        dax::testing::Testing::CellCheckHexahedron());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletSlice(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestSlice);
}
