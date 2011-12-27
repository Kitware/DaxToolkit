/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/WorkMapField.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <algorithm>
#include <vector>

#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

static void TestMapFieldVoxel(
  dax::exec::WorkMapField<dax::exec::CellVoxel> work,
  const dax::internal::StructureUniformGrid &gridstruct,
  dax::Id pointFlatIndex)
{
  if (work.GetIndex() != pointFlatIndex)
    {
    TEST_FAIL(<< "Work object returned wrong index.");
    }

  dax::Id3 pointIjkIndex = dax::internal::flatIndexToIndex3(pointFlatIndex,
                                                            gridstruct.Extent);

  dax::Id3 dim = dax::internal::extentDimensions(gridstruct.Extent);
  dax::Id numPoints = dim[0]*dim[1]*dim[2];

  std::vector<dax::Scalar> fieldData(numPoints);
  std::fill(fieldData.begin(), fieldData.end(), -1.0);
  fieldData[pointFlatIndex] = pointFlatIndex;

  dax::internal::DataArray<dax::Scalar> fieldArray(&fieldData.at(0),
                                                   fieldData.size());
  dax::exec::FieldPoint<dax::Scalar> field(fieldArray);

  dax::Scalar scalarValue = work.GetFieldValue(field);
  if (scalarValue != pointFlatIndex)
    {
    TEST_FAIL(<< "Did not get expected data value.");
    }

  work.SetFieldValue(field, static_cast<dax::Scalar>(-2));
  if (fieldData[pointFlatIndex] != -2)
    {
    TEST_FAIL(<< "Field value did not set as expected.");
    }

  dax::Vector3 expectedCoords
      = dax::make_Vector3(static_cast<dax::Scalar>(pointIjkIndex[0]),
                          static_cast<dax::Scalar>(pointIjkIndex[1]),
                          static_cast<dax::Scalar>(pointIjkIndex[2]));
  expectedCoords = gridstruct.Origin + expectedCoords * gridstruct.Spacing;

  dax::internal::DataArray<dax::Vector3> dummyArray;
  dax::exec::FieldCoordinates fieldCoords(dummyArray);
  dax::Vector3 coords = work.GetFieldValue(fieldCoords);

  if (expectedCoords != coords)
    {
    TEST_FAIL(<< "Did not get expected point coordinates.");
    }
}

static void TestMapFieldVoxel()
{
  std::cout << "Testing WorkMapField<CellVoxel>" << std::endl;

  dax::internal::StructureUniformGrid gridstruct;

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(9, 9, 9);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::WorkMapField<dax::exec::CellVoxel> work(gridstruct, flatIndex);
    TestMapFieldVoxel(work, gridstruct, flatIndex);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(14, 5, 12);
  dax::exec::WorkMapField<dax::exec::CellVoxel> work(gridstruct, 0);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    work.SetIndex(flatIndex);
    TestMapFieldVoxel(work, gridstruct, flatIndex);
    }
}

int UnitTestWorkMapField(int, char *[])
{
  try
    {
    TestMapFieldVoxel();
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
