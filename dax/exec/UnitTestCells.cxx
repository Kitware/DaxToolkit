/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <dax/exec/Cell.h>

#include <dax/internal/Testing.h>
#include <vector>

namespace
{
static void CheckPointIndex(dax::Id pointFlatIndex,
                            const dax::Id3 &pointIjkIndex,
                            const dax::Extent3 &extent)
{
  dax::Id3 compareIndex = dax::flatIndexToIndex3(pointFlatIndex, extent);

  DAX_TEST_ASSERT(compareIndex == pointIjkIndex,
                  "Bad point index.");
}

static void CheckPointIndex(const dax::Id &hexPointIndex,
                            const dax::Id &voxPointIndex)
{
  DAX_TEST_ASSERT(hexPointIndex == voxPointIndex,
                  "Bad point index.");
}
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellHexahedron.
dax::internal::UnstructuredGrid<dax::exec::CellHexahedron>
  make_ugrid(const dax::internal::TopologyUniformGrid& uniform,
             std::vector<dax::Vector3>& points,
             std::vector<dax::Id>& topology
             )
{
  //copy the point info over to the unstructured grid
  points.clear();
  for(dax::Id i=0; i <dax::internal::numberOfPoints(uniform); ++i)
    {
    points.push_back(dax::internal::pointCoordiantes(uniform,i));
    }

  //copy the cell topology information over
  topology.clear();
  for(dax::Id i=0; i <dax::internal::numberOfCells(uniform); ++i)
    {
    dax::exec::CellVoxel vox(uniform,i);
    for(dax::Id j=0; j < vox.GetNumberOfPoints(); ++j)
      {
      topology.push_back(vox.GetPointIndex(j));
      }
    }

  dax::internal::DataArray<dax::Vector3> rawPoints(&points[0],points.size());
  dax::internal::DataArray<dax::Id> rawTopo(&topology[0],topology.size());
  dax::internal::UnstructuredGrid<dax::exec::CellHexahedron> ugrid(rawPoints,
                                                                   rawTopo);
  return ugrid;
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellVoxel.
void TestCellVoxel(const dax::exec::CellVoxel cell,
                   const dax::internal::TopologyUniformGrid &gridstruct,
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

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellHexahedron.
void TestCellHexahedron(const dax::exec::CellHexahedron cell,
                        const dax::exec::CellVoxel voxel)
{
  DAX_TEST_ASSERT(cell.GetNumberOfPoints() == voxel.GetNumberOfPoints(),
                 "CellHexahedron has wrong number of points");

  DAX_TEST_ASSERT(cell.GetIndex() == voxel.GetIndex(),
                  "CellHexahedron has different index for cell");

  CheckPointIndex(cell.GetPointIndex(0), voxel.GetPointIndex(0));
  CheckPointIndex(cell.GetPointIndex(1), voxel.GetPointIndex(1));
  CheckPointIndex(cell.GetPointIndex(2), voxel.GetPointIndex(2));
  CheckPointIndex(cell.GetPointIndex(3), voxel.GetPointIndex(3));
  CheckPointIndex(cell.GetPointIndex(4), voxel.GetPointIndex(4));
  CheckPointIndex(cell.GetPointIndex(5), voxel.GetPointIndex(5));
  CheckPointIndex(cell.GetPointIndex(6), voxel.GetPointIndex(6));
  CheckPointIndex(cell.GetPointIndex(7), voxel.GetPointIndex(7));
}

static void TestCellVoxel()
{
  dax::internal::TopologyUniformGrid gridstruct;

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

static void TestCellHexahedron()
{
  std::vector<dax::Id> topo;
  std::vector<dax::Vector3> points;
  dax::internal::TopologyUniformGrid gridstruct;
  dax::internal::UnstructuredGrid<dax::exec::CellHexahedron> ugrid;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  ugrid = make_ugrid(gridstruct,points,topo);

  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::CellHexahedron hex(ugrid, flatIndex);
    dax::exec::CellVoxel vox(gridstruct,flatIndex);
    TestCellHexahedron(hex,vox);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  ugrid = make_ugrid(gridstruct,points,topo);

  dax::exec::CellHexahedron cell(ugrid, 0);
  dax::exec::CellVoxel vox(gridstruct,0);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    cell.SetIndex(flatIndex);
    vox.SetIndex(flatIndex);
    TestCellHexahedron(cell,vox);
    }
}

static void TestCells()
{
  TestCellVoxel();
  TestCellHexahedron();
}

int UnitTestCells(int, char *[])
{
  return dax::internal::Testing::Run(TestCells);
}
