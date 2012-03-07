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

#include <dax/Extent.h>

#include <dax/internal/Testing.h>

namespace {

//-----------------------------------------------------------------------------
static void TestDimensions()
{
  std::cout << "Testing Dimensions" << std::endl;

  dax::Extent3 extent;
  dax::Id3 dims;

  extent.Min = dax::make_Id3(0, 0, 0);
  extent.Max = dax::make_Id3(10, 10, 10);
  dims = dax::extentDimensions(extent);
  DAX_TEST_ASSERT((dims[0] == 11) && (dims[1] == 11) && (dims[2] == 11),
                  "Got incorrect dimensions for extent.");

  extent.Min = dax::make_Id3(-5, 8, 23);
  extent.Max = dax::make_Id3(10, 25, 44);
  dims = dax::extentDimensions(extent);
  DAX_TEST_ASSERT((dims[0] == 16) && (dims[1] == 18) && (dims[2] == 22),
                  "Got incorrect dimensions for extent.");
}

//-----------------------------------------------------------------------------
static void TestIndexConversion(dax::Extent3 extent)
{
  dax::Id3 dims = dax::extentDimensions(extent);
  dax::Id correctFlatIndex;
  dax::Id3 correctIndex3;

  std::cout << "Testing point index conversion" << std::endl;
  correctFlatIndex = 0;
  for (correctIndex3[2] = extent.Min[2];
       correctIndex3[2] <= extent.Max[2];
       correctIndex3[2]++)
    {
    for (correctIndex3[1] = extent.Min[1];
         correctIndex3[1] <= extent.Max[1];
         correctIndex3[1]++)
      {
      for (correctIndex3[0] = extent.Min[0];
           correctIndex3[0] <= extent.Max[0];
           correctIndex3[0]++)
        {
        dax::Id computedFlatIndex
            = dax::index3ToFlatIndex(correctIndex3, extent);
        DAX_TEST_ASSERT(computedFlatIndex == correctFlatIndex,
                        "Got incorrect flat index");

        dax::Id3 computedIndex3
            = dax::flatIndexToIndex3(correctFlatIndex, extent);
        DAX_TEST_ASSERT(   (computedIndex3[0] == correctIndex3[0])
                        && (computedIndex3[1] == correctIndex3[1])
                        && (computedIndex3[2] == correctIndex3[2]),
                        "Got incorrect 3d index");

        correctFlatIndex++;
        }
      }
    }
  DAX_TEST_ASSERT(correctFlatIndex == dims[0]*dims[1]*dims[2],
                  "Tested wrong number of indices.");

  std::cout << "Testing cell index conversion" << std::endl;
  correctFlatIndex = 0;
  for (correctIndex3[2] = extent.Min[2];
       correctIndex3[2] < extent.Max[2];
       correctIndex3[2]++)
    {
    for (correctIndex3[1] = extent.Min[1];
         correctIndex3[1] < extent.Max[1];
         correctIndex3[1]++)
      {
      for (correctIndex3[0] = extent.Min[0];
           correctIndex3[0] < extent.Max[0];
           correctIndex3[0]++)
        {
        dax::Id computedFlatIndex
            = dax::index3ToFlatIndexCell(correctIndex3, extent);
        DAX_TEST_ASSERT(computedFlatIndex == correctFlatIndex,
                        "Got incorrect flat index");

        dax::Id3 computedIndex3
            = dax::flatIndexToIndex3Cell(correctFlatIndex, extent);
        DAX_TEST_ASSERT(   (computedIndex3[0] == correctIndex3[0])
                        && (computedIndex3[1] == correctIndex3[1])
                        && (computedIndex3[2] == correctIndex3[2]),
                        "Got incorrect 3d index");

        correctFlatIndex++;
        }
      }
    }
  DAX_TEST_ASSERT(correctFlatIndex == (dims[0]-1)*(dims[1]-1)*(dims[2]-1),
                  "Tested wrong number of indices.");
}

//-----------------------------------------------------------------------------
static void TestIndexConversion()
{
  std::cout << "Testing index converstion." << std::endl;

  dax::Extent3 extent;

  extent.Min = dax::make_Id3(0, 0, 0);
  extent.Max = dax::make_Id3(10, 10, 10);
  TestIndexConversion(extent);

  extent.Min = dax::make_Id3(-5, 8, 23);
  extent.Max = dax::make_Id3(10, 25, 44);
  TestIndexConversion(extent);
}

static void ExtentTests()
{
  TestDimensions();
  TestIndexConversion();
}

} // anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestExtent(int, char *[])
{
  return dax::internal::Testing::Run(ExtentTests);
}
