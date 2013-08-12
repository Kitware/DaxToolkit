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

#include <dax/worklet/Tetrahedralize.h>

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
#include <dax/cont/testing/Testing.h>



namespace {
const dax::Id DIM = 26;

void verify_cell_values_written(std::vector<dax::Id>& data)
{
  typedef std::vector<dax::Id>::iterator it;
  for( it i = data.begin(); i != data.end(); ++i)
    {
    DAX_TEST_ASSERT(*i!=-1,"didn't write into cell connections");
    }
}

void verify_cell_values_correct(std::vector<dax::Id>& data, dax::Id numPoints)
{
  typedef std::vector<dax::Id>::iterator it;

  std::vector<dax::Id> counts(numPoints,0);
  for( it i = data.begin(); i != data.end(); ++i)
    {
    ++counts[*i];
    }

  for( it i = counts.begin(); i != counts.end(); ++i)
    {
    bool valid(*i == 1 || *i==2 || *i == 4 || *i == 8 || *i == 16 || *i == 32);
    DAX_TEST_ASSERT(valid==true,"didn't write valid topology");
    }

}

//-----------------------------------------------------------------------------
struct TestTetrahedralizeWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::testing::TestGrid<GridType> in(DIM);
    GridType out;

    this->GridTetrahedralize(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  void operator()(const dax::cont::UniformGrid<>&) const
    {
    dax::cont::testing::TestGrid<dax::cont::UniformGrid<> > in(DIM);
    dax::cont::UnstructuredGrid<dax::CellTagTetrahedron> out;

    this->GridTetrahedralize(in.GetRealGrid(),out);
    }

  //----------------------------------------------------------------------------
  template <typename InGridType,
            typename OutGridType>
  void GridTetrahedralize(const InGridType& inGrid, OutGridType& outGrid) const
    {
    const dax::Id cellConnLength = inGrid.GetNumberOfCells() * 5 * 4 ;
    std::vector<dax::Id> cellConnections(cellConnLength,-1);
    dax::cont::ArrayHandle<dax::Id> cellHandle =
            dax::cont::make_ArrayHandle(cellConnections);
    try
      {
      typedef dax::cont::GenerateTopology<dax::worklet::Tetrahedralize,
                                          dax::cont::ArrayHandleConstant<dax::Id>
                                          > GenerateT;
      typedef typename GenerateT::ClassifyResultType  ClassifyResultType;

      dax::cont::Scheduler<> scheduler;
      ClassifyResultType classification(5,inGrid.GetNumberOfCells());

      dax::worklet::Tetrahedralize tetWorklet;
      GenerateT generateTets(classification,tetWorklet);

      //don't remove duplicate points.
      generateTets.SetRemoveDuplicatePoints(false);

      outGrid.SetCellConnections(cellHandle);
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

    cellHandle.CopyInto(cellConnections.begin());
    verify_cell_values_written(cellConnections);
    verify_cell_values_correct(cellConnections,inGrid.GetNumberOfPoints());
    }
};


//-----------------------------------------------------------------------------
void TestTetrahedralize()
  {
  // TODO: We should support more tetrahedralization than voxels, and we should
  // test that, too.
  dax::cont::testing::GridTesting::TryAllGridTypes(
        TestTetrahedralizeWorklet(),
        dax::testing::Testing::CellCheckUniform());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletTetrahedralize(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestTetrahedralize);
}
