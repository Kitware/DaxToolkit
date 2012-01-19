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

int UnitTestGridStructures(int, char *[])
{
  try
    {
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
