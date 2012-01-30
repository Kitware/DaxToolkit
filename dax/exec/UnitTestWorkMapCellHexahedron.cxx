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

extern void TestCellHexahedron(const dax::exec::CellHexahedron cell,
                               const dax::exec::CellHexahedron Hexahedron);

extern dax::internal::UnstructuredGrid<dax::exec::CellHexahedron> make_ugrid(
    const dax::internal::StructureUniformGrid& uniform,
    std::vector<dax::Vector3>& points,
    std::vector<dax::Id>& topology);


#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

static void TestMapCellHexahedron(
  dax::exec::WorkMapCell<dax::exec::CellHexahedron> &work,
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

  TestCellHexahedron(work.GetCell(), gridstruct, cellFlatIndex);
}

static void TestMapCellHexahedron()
{
  std::cout << "Testing WorkMapCell<CellHexahedron>" << std::endl;

  dax::internal::StructureUniformGrid gridstruct;

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfCells(gridstruct);
       flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellHexahedron> work(gridstruct, flatIndex);
    TestMapCellHexahedron(work, gridstruct, flatIndex);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  dax::exec::WorkMapCell<dax::exec::CellHexahedron> work(gridstruct, 0);
  for (dax::Id flatIndex = 0;
       flatIndex < dax::internal::numberOfPoints(gridstruct);
       flatIndex++)
    {
    work.SetCellIndex(flatIndex);
    TestMapCellHexahedron(work, gridstruct, flatIndex);
    }
}

int UnitTestWorkMapCellHexahedron(int, char *[])
{
  try
    {
    TestMapCellHexahedron();
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
