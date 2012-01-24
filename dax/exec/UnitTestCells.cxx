/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/Cell.h>

#include <dax/internal/Testing.h>

static void CheckPointIndex(dax::Id pointFlatIndex,
                            const dax::Id3 &pointIjkIndex,
                            const dax::Extent3 &extent)
{
  dax::Id3 compareIndex = dax::flatIndexToIndex3(pointFlatIndex, extent);

  DAX_TEST_ASSERT(compareIndex == pointIjkIndex,
                  "Bad point index.");
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCell.
void TestCellVoxel(const dax::exec::CellVoxel cell,
                   const dax::internal::StructureUniformGrid &gridstruct,
                   dax::Id cellFlatIndex)
{
  dax::Id3 cellIjkIndex
      = dax::flatIndexToIndex3Cell(cellFlatIndex, gridstruct.Extent);

  DAX_TEST_ASSERT(cell.GetNumberOfPoints() == 8,
                  "CellVoxel has wrong number of points");

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

  DAX_TEST_ASSERT(cell.GetOrigin() == gridstruct.Origin,
                  "CellVoxel has wrong origin");

  DAX_TEST_ASSERT(cell.GetSpacing() == gridstruct.Spacing,
                  "CellVoxel has wrong spacing");

  DAX_TEST_ASSERT(cell.GetExtent().Min == gridstruct.Extent.Min,
                  "CellVoxel has wrong extent");
  DAX_TEST_ASSERT(cell.GetExtent().Max == gridstruct.Extent.Max,
                  "CellVoxel has wrong extent");

  DAX_TEST_ASSERT(cell.GetIndex() == cellFlatIndex,
                  "CellVoxel has wrong index");
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

static void TestCells()
{
  TestCellVoxel();
}

int UnitTestCells(int, char *[])
{
  return dax::internal::Testing::Run(TestCells);
}
