/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_internal_TestingGridGenerator_h
#define  __dax_cont_internal_TestingGridGenerator_h

#include <dax/Types.h>

#include <dax/exec/Cell.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

namespace dax {
namespace cont {
namespace internal {

template<typename T, typename DA>
struct TestGrid
{
  typedef T GridType;
  typedef DA DeviceAdapter;

  GridType Grid;
  const dax::Id Size;

  TestGrid(const dax::Id& size)
    :Size(size),Grid()
    {
    this->gridInfo = GridBuilder(this->Grid,size);
    }

  // Description:
  // Enable pointer-like dereference syntax. Returns a pointer to the contained
  // object.
  const GridType* operator->() const
    {
    return &this->Grid;
    }

  // Description:
  // Get a raw pointer to the contained object.
  const GridType& GetRealGrid() const
    {
    return this->Grid;
    }

private:
  //helper structs needed to have worklet tests work
  //on both uniform and unstructured grids
  struct GridBuilder
  {
    GridBuilder()
      {

      }

    GridBuilder(dax::cont::UniformGrid &grid, const dax::Id& DIM)
      {
      grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));
      }

    GridBuilder(dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> &grid, const dax::Id& DIM)
      {
      //we need to make a volume grid
      dax::cont::UniformGrid uniform;
      uniform.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

      //copy the point info over to the unstructured grid
      points.clear();
      for(dax::Id i=0; i <uniform.GetNumberOfPoints(); ++i)
        {
        points.push_back(uniform.GetPointCoordinates(i));
        }

      //copy the cell topology information over

      //this only works for voxel/hexahedron
      const dax::Id3 cellVertexToPointIndex[8] = {
        dax::make_Id3(0, 0, 0),
        dax::make_Id3(1, 0, 0),
        dax::make_Id3(1, 1, 0),
        dax::make_Id3(0, 1, 0),
        dax::make_Id3(0, 0, 1),
        dax::make_Id3(1, 0, 1),
        dax::make_Id3(1, 1, 1),
        dax::make_Id3(0, 1, 1)
      };

      topology.clear();
      dax::Id numPointsPerCell = ::dax::exec::CellHexahedron::NUM_POINTS;
      const dax::Extent3 extents = uniform.GetExtent();
      for(dax::Id i=0; i <uniform.GetNumberOfCells(); ++i)
        {
        dax::Id3 ijkCell = dax::flatIndexToIndex3Cell(i,extents);
        for(dax::Id j=0; j < numPointsPerCell; ++j)
          {
          dax::Id3 ijkPoint = ijkCell + cellVertexToPointIndex[j];

          dax::Id pointIndex = index3ToFlatIndex(ijkPoint,extents);
          topology.push_back(pointIndex);
          }
        }

      dax::cont::ArrayHandle<dax::Vector3,DeviceAdapter> ahPoints(points.begin(),points.end());
      dax::cont::ArrayHandle<dax::Id,DeviceAdapter> ahTopo(topology.begin(),topology.end());

      grid.UpdateHandles(ahTopo,ahPoints);
      }

    std::vector<dax::Id> topology;
    std::vector<dax::Vector3> points;
  };
  GridBuilder gridInfo;
};

}
}
} //namespace dax::cont::internal

#endif //  __dax_cont_internal_TestingGridGenerator_h
