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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

#include <dax/worklet/Tetrahedralize.worklet>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <dax/CellTag.h>
#include <dax/TypeTraits.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/GenerateTopology.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/internal/testing/Testing.h>



namespace {
const dax::Id DIM = 26;

//-----------------------------------------------------------------------------
struct TestTetrahedralizeWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::internal::TestGrid<GridType> in(DIM);
    GridType out;

    this->GridTetrahedralize(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::CellTagTetrahedron> out;

    this->GridTetrahedralize(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridTetrahedralize(const InGridType& inGrid, OutGridType& outGrid) const
    {
    try
      {
      typedef dax::cont::GenerateTopology<dax::worklet::Tetrahedralize,
                                          dax::cont::ArrayHandleConstantValue<dax::Id>
                                          > GenerateT;
      typedef typename GenerateT::ClassifyResultType  ClassifyResultType;

      dax::cont::Scheduler<> scheduler;
      ClassifyResultType classification(5,inGrid.GetNumberOfCells());

      dax::worklet::Tetrahedralize tetWorklet;
      GenerateT generateTets(classification,tetWorklet);

      //don't remove duplicate points.
      generateTets.SetRemoveDuplicatePoints(false);

      scheduler.Invoke(generateTets,inGrid,outGrid);

      }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
    DAX_TEST_ASSERT(outGrid.GetNumberOfCells()==inGrid.GetNumberOfCells() * 5,
                    "Incorrect number of cells in the output grid");

    //should have zero points since we didn't assign a points coordinate array
    //to the unstructured grid
    DAX_TEST_ASSERT(outGrid.GetNumberOfPoints()==0,
                    "Incorrect number of points in the output grid");
    }
};


//-----------------------------------------------------------------------------
void TestTetrahedralize()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestTetrahedralizeWorklet(),
                     dax::cont::internal::GridTesting::TypeCheckUniformGrid());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletTetrahedralize(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestTetrahedralize);
}
