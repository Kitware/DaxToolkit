/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/WorkMapCell.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <algorithm>
#include <vector>

extern void TestCellVoxel(const dax::exec::CellVoxel cell,
                          const dax::internal::StructureUniformGrid &gridstruct,
                          dax::Id cellFlatIndex);


#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler ErrorHandler
  = dax::exec::internal::ErrorHandler(dax::internal::DataArray<char>());

}  // Anonymous namespace

static void TestMapCellVoxel(
  dax::exec::WorkMapCell<dax::exec::CellVoxel> &work,
  const dax::internal::StructureUniformGrid &gridstruct,
  dax::Id cellFlatIndex)
{
  if (work.GetCellIndex() != cellFlatIndex)
    {
    TEST_FAIL(<< "Work object returned wrong cell index.");
    }

  dax::Id3 dim = dax::extentDimensions(gridstruct.Extent);
  dax::Id numCells = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numCells);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[cellFlatIndex] = cellFlatIndex;

  dax::internal::DataArray<dax::Scalar> fieldArray(&fieldData.at(0),
                                                   fieldData.size());
  dax::exec::FieldCell<dax::Scalar> field(fieldArray);

  dax::Scalar scalarValue = work.GetFieldValue(field);
  if (scalarValue != cellFlatIndex)
    {
    TEST_FAIL(<< "Did not get expected data value.");
    }

  work.SetFieldValue(field, static_cast<dax::Scalar>(-2));
  if (fieldData[cellFlatIndex] != -2)
    {
    TEST_FAIL(<< "Field value did not set as expected.");
    }

  TestCellVoxel(work.GetCell(), gridstruct, cellFlatIndex);
}

static void TestMapCellVoxel()
{
  std::cout << "Testing WorkMapCell<CellVoxel>" << std::endl;

  {
  dax::internal::StructureUniformGrid gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  dax::exec::WorkMapCell<dax::exec::CellVoxel> work(gridstruct, ErrorHandler);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    work.SetCellIndex(flatIndex);
    TestMapCellVoxel(work, gridstruct, flatIndex);
    }
  }

  {
  dax::internal::StructureUniformGrid gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  dax::exec::WorkMapCell<dax::exec::CellVoxel> work(gridstruct, ErrorHandler);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfPoints(gridstruct);
       flatIndex++)
    {
    work.SetCellIndex(flatIndex);
    TestMapCellVoxel(work, gridstruct, flatIndex);
    }
  }
}

int UnitTestWorkMapCell(int, char *[])
{
  try
    {
    TestMapCellVoxel();
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
