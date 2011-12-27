/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/ParametricCoordinates.h>

#include <dax/exec/WorkMapCell.h>

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

template<class WorkType, class CellType>
static void CompareCoordinates(const WorkType &work,
                               const CellType &cell,
                               const dax::exec::FieldCoordinates &coordField,
                               dax::Vector3 truePCoords,
                               dax::Vector3 trueWCoords)
{
  dax::Vector3 computedWCoords
      = dax::exec::parametricCoordinatesToWorldCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           truePCoords);
  if (computedWCoords != trueWCoords)
    {
    TEST_FAIL(<< "Computed wrong world coords from parametric coords.");
    }

  dax::Vector3 computedPCoords
      = dax::exec::worldCoordinatesToParametricCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           trueWCoords);
  if (computedPCoords != truePCoords)
    {
    TEST_FAIL(<< "Computed wrong parametric coords from world coords.");
    }
}

static void TestPCoordsVoxel(
  const dax::exec::WorkMapCell<dax::exec::CellVoxel> &work,
  const dax::exec::FieldCoordinates &coordField)
{
  const dax::Vector3 cellVertexToParametricCoords[8] = {
    dax::make_Vector3(0, 0, 0),
    dax::make_Vector3(1, 0, 0),
    dax::make_Vector3(1, 1, 0),
    dax::make_Vector3(0, 1, 0),
    dax::make_Vector3(0, 0, 1),
    dax::make_Vector3(1, 0, 1),
    dax::make_Vector3(1, 1, 1),
    dax::make_Vector3(0, 1, 1)
  };
  const dax::exec::CellVoxel &cell = work.GetCell();

  // Check the coordinates at all vertices
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 truePCoords = cellVertexToParametricCoords[vertexIndex];
    dax::Vector3 trueWCoords = work.GetFieldValue(coordField, vertexIndex);
    CompareCoordinates(work, cell, coordField, truePCoords, trueWCoords);
    }

  dax::Vector3 centerCoords = dax::make_Vector3(0.0, 0.0, 0.0);
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    centerCoords = centerCoords + work.GetFieldValue(coordField, vertexIndex);
    }
  centerCoords = (1.0/8.0) * centerCoords;
  CompareCoordinates(
        work, cell, coordField, dax::make_Vector3(0.5,0.5,0.5), centerCoords);
}

static void TestPCoordsVoxel()
{
  std::cout << "Testing TestPCoords<CellVoxel>" << std::endl;

  dax::internal::StructureUniformGrid gridstruct;

  dax::internal::DataArray<dax::Vector3> dummyArray;
  dax::exec::FieldCoordinates coordField(dummyArray);

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellVoxel> work(gridstruct, flatIndex);
    TestPCoordsVoxel(work, coordField);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  dax::exec::WorkMapCell<dax::exec::CellVoxel> work(gridstruct, 0);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    work.SetCellIndex(flatIndex);
    TestPCoordsVoxel(work, coordField);
    }
}

int UnitTestParametricCoordinates(int, char *[])
{
  try
    {
    TestPCoordsVoxel();
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
