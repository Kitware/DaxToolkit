/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/Cell.h>

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

static void CheckPointIndex(dax::Id pointFlatIndex,
                            const dax::Id3 &pointIjkIndex,
                            const dax::internal::Extent3 &extent)
{
  dax::Id3 compareIndex = dax::internal::flatIndexToIndex3(pointFlatIndex,
                                                           extent);

  if (compareIndex != pointIjkIndex)
    {
    TEST_FAIL(<<"Bad point index.");
    }
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCell.
void TestCellVoxel(const dax::exec::CellVoxel cell,
                   const dax::internal::StructureUniformGrid &gridstruct,
                   dax::Id cellFlatIndex)
{
  dax::Id3 cellIjkIndex
      = dax::internal::flatIndexToIndex3Cell(cellFlatIndex, gridstruct.Extent);

  if (cell.GetNumberOfPoints() != 8)
    {
    TEST_FAIL(<< "CellVoxel has wrong number of points");
    }

  CheckPointIndex(cell.GetPointIndex(0), cellIjkIndex, gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(1),
                  cellIjkIndex + dax::make_Id3(1,0,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(2),
                  cellIjkIndex + dax::make_Id3(1,1,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(3),
                  cellIjkIndex + dax::make_Id3(0,1,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(4),
                  cellIjkIndex + dax::make_Id3(0,0,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(5),
                  cellIjkIndex + dax::make_Id3(1,0,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(6),
                  cellIjkIndex + dax::make_Id3(1,1,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(7),
                  cellIjkIndex + dax::make_Id3(0,1,1),
                  gridstruct.Extent);

  if (cell.GetOrigin() != gridstruct.Origin)
    {
    TEST_FAIL(<< "CellVoxel has wrong origin");
    }

  if (cell.GetSpacing() != gridstruct.Spacing)
    {
    TEST_FAIL(<< "CellVoxel has wrong spacing");
    }

  if (   (cell.GetExtent().Min != gridstruct.Extent.Min)
      || (cell.GetExtent().Max != gridstruct.Extent.Max) )
    {
    TEST_FAIL(<< "CellVoxel has wrong extent");
    }

  if (cell.GetIndex() != cellFlatIndex)
    {
    TEST_FAIL(<< "CellVoxel has wrong index");
    }
}

static void TestCellVoxel()
{
  dax::internal::StructureUniformGrid gridstruct;

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::CellVoxel cell(gridstruct, flatIndex);
    TestCellVoxel(cell, gridstruct, flatIndex);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  dax::exec::CellVoxel cell(gridstruct, 0);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    cell.SetIndex(flatIndex);
    TestCellVoxel(cell, gridstruct, flatIndex);
    }
}

int UnitTestCells(int, char *[])
{
  try
    {
    TestCellVoxel();
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
