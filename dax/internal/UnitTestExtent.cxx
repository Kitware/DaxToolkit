/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/internal/GridStructures.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

static void TestDimensions()
{
  std::cout << "Testing Dimensions" << std::endl;

  dax::internal::Extent3 extent;
  dax::Id3 dims;

  extent.Min = dax::make_Id3(0, 0, 0);
  extent.Max = dax::make_Id3(10, 10, 10);
  dims = dax::internal::extentDimensions(extent);
  if ((dims.x != 11) || (dims.y != 11) || (dims.z != 11))
    {
    TEST_FAIL(<< "Got incorrect dimensions for extent.");
    }

  extent.Min = dax::make_Id3(-5, 8, 23);
  extent.Max = dax::make_Id3(10, 25, 44);
  dims = dax::internal::extentDimensions(extent);
  if ((dims.x != 16) || (dims.y != 18) || (dims.z != 22))
    {
    TEST_FAIL(<< "Got incorrect dimensions for extent.");
    }
}

static void TestIndexConversion(dax::internal::Extent3 extent)
{
  dax::Id3 dims = dax::internal::extentDimensions(extent);
  dax::Id correctFlatIndex;
  dax::Id3 correctIndex3;

  std::cout << "Testing point index conversion" << std::endl;
  correctFlatIndex = 0;
  for (correctIndex3.z = extent.Min.z;
       correctIndex3.z <= extent.Max.z;
       correctIndex3.z++)
    {
    for (correctIndex3.y = extent.Min.y;
         correctIndex3.y <= extent.Max.y;
         correctIndex3.y++)
      {
      for (correctIndex3.x = extent.Min.x;
           correctIndex3.x <= extent.Max.x;
           correctIndex3.x++)
        {
        dax::Id computedFlatIndex
            = dax::internal::index3ToFlatIndex(correctIndex3, extent);
        if (computedFlatIndex != correctFlatIndex)
          {
          TEST_FAIL(<< "Got incorrect flat index");
          }

        dax::Id3 computedIndex3
            = dax::internal::flatIndexToIndex3(correctFlatIndex, extent);
        if (   (computedIndex3.x != correctIndex3.x)
            || (computedIndex3.y != correctIndex3.y)
            || (computedIndex3.z != correctIndex3.z) )
          {
          TEST_FAIL(<< "Got incorrect 3d index");
          }

        correctFlatIndex++;
        }
      }
    }
  if (correctFlatIndex != dims.x*dims.y*dims.z)
    {
    TEST_FAIL(<< "Tested wrong number of indices.");
    }

  std::cout << "Testing cell index conversion" << std::endl;
  correctFlatIndex = 0;
  for (correctIndex3.z = extent.Min.z;
       correctIndex3.z < extent.Max.z;
       correctIndex3.z++)
    {
    for (correctIndex3.y = extent.Min.y;
         correctIndex3.y < extent.Max.y;
         correctIndex3.y++)
      {
      for (correctIndex3.x = extent.Min.x;
           correctIndex3.x < extent.Max.x;
           correctIndex3.x++)
        {
        dax::Id computedFlatIndex
            = dax::internal::index3ToFlatIndexCell(correctIndex3, extent);
        if (computedFlatIndex != correctFlatIndex)
          {
          TEST_FAIL(<< "Got incorrect flat index");
          }

        dax::Id3 computedIndex3
            = dax::internal::flatIndexToIndex3Cell(correctFlatIndex, extent);
        if (   (computedIndex3.x != correctIndex3.x)
            || (computedIndex3.y != correctIndex3.y)
            || (computedIndex3.z != correctIndex3.z) )
          {
          TEST_FAIL(<< "Got incorrect 3d index");
          }

        correctFlatIndex++;
        }
      }
    }
  if (correctFlatIndex != (dims.x-1)*(dims.y-1)*(dims.z-1))
    {
    TEST_FAIL(<< "Tested wrong number of indices.");
    }
}

static void TestIndexConversion()
{
  std::cout << "Testing index converstion." << std::endl;

  dax::internal::Extent3 extent;

  extent.Min = dax::make_Id3(0, 0, 0);
  extent.Max = dax::make_Id3(10, 10, 10);
  TestIndexConversion(extent);

  extent.Min = dax::make_Id3(-5, 8, 23);
  extent.Max = dax::make_Id3(10, 25, 44);
  TestIndexConversion(extent);
}

static void TestGridSize(const dax::internal::StructureUniformGrid &gridstruct,
                         dax::Id numPoints,
                         dax::Id numCells)
{
  dax::Id computedNumPoints = dax::internal::numberOfPoints(gridstruct);
  if (computedNumPoints != numPoints)
    {
    TEST_FAIL(<< "Structured grid returned wrong number of points");
    }

  dax::Id computedNumCells = dax::internal::numberOfCells(gridstruct);
  if (computedNumCells != numCells)
    {
    TEST_FAIL(<< "Structured grid return wrong number of cells");
    }
}

static void TestGridSize()
{
  std::cout << "Testing grid size." << std::endl;

  dax::internal::StructureUniformGrid gridstruct;
  gridstruct.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);

  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  TestGridSize(gridstruct, 1331, 1000);

  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  TestGridSize(gridstruct, 1936, 1500);
}

int UnitTestExtent(int, char *[])
{
  try
    {
    TestDimensions();
    TestIndexConversion();
    TestGridSize();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }

  return 0;
}
