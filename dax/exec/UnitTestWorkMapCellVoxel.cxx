/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/WorkMapCell.h>

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

extern void TestCellVoxel(const dax::exec::CellVoxel cell,
                          const dax::internal::TopologyUniformGrid &gridstruct,
                          dax::Id cellFlatIndex);


namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler ErrorHandler
  = dax::exec::internal::ErrorHandler(dax::internal::DataArray<char>());

}  // Anonymous namespace

static void TestMapCellVoxel(
  dax::exec::WorkMapCell<dax::exec::CellVoxel> &work,
  const dax::internal::TopologyUniformGrid &gridstruct,
  dax::Id cellFlatIndex)
{
  DAX_TEST_ASSERT(work.GetCellIndex() == cellFlatIndex,
                  "Work object returned wrong cell index.");

  dax::Id3 dim = dax::extentDimensions(gridstruct.Extent);
  dax::Id numCells = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numCells);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[cellFlatIndex] = cellFlatIndex;

  dax::internal::DataArray<dax::Scalar> fieldArray(&fieldData.at(0),
                                                   fieldData.size());
  dax::exec::FieldCell<dax::Scalar> field(fieldArray);

  dax::Scalar scalarValue = work.GetFieldValue(field);
  DAX_TEST_ASSERT(scalarValue == cellFlatIndex,
                  "Did not get expected data value.");

  work.SetFieldValue(field, static_cast<dax::Scalar>(-2));
  DAX_TEST_ASSERT(fieldData[cellFlatIndex] == -2,
                  "Field value did not set as expected.");

  TestCellVoxel(work.GetCell(), gridstruct, cellFlatIndex);
}

static void TestMapCellVoxel()
{
  std::cout << "Testing WorkMapCell<CellVoxel>" << std::endl;

  {
  dax::internal::TopologyUniformGrid gridstruct;
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
  dax::internal::TopologyUniformGrid gridstruct;
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

static void TestMapCell()
{
  TestMapCellVoxel();
}

int UnitTestWorkMapCellVoxel(int, char *[])
{
  return dax::internal::Testing::Run(TestMapCell);
}
