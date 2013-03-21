//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#ifndef __MarchingCubes_worklet_
#define __MarchingCubes_worklet_

#include <dax/CellTag.h>
#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/InterpolatedCellPoints.h>
#include <dax/exec/WorkletInterpolatedCell.h>
#include <dax/exec/WorkletMapCell.h>

#include <dax/worklet/internal/MarchingCubesTable.h>

namespace dax {
namespace worklet {

// -----------------------------------------------------------------------------
template<typename T, typename U>
DAX_EXEC_EXPORT
int GetVoxelClassification(const T isoValue, const U& values )
{
  return ((values[0] > isoValue) << 0 |
          (values[1] > isoValue) << 1 |
          (values[2] > isoValue) << 2 |
          (values[3] > isoValue) << 3 |
          (values[4] > isoValue) << 4 |
          (values[5] > isoValue) << 5 |
          (values[6] > isoValue) << 6 |
          (values[7] > isoValue) << 7);
}

// -----------------------------------------------------------------------------
class MarchingCubesClassify : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(Point), Field(Out));
  typedef _3 ExecutionSignature(_2);

  DAX_CONT_EXPORT MarchingCubesClassify(dax::Scalar isoValue)
    : IsoValue(isoValue) {  }

  // TODO: This method should be overloaded such that it accepts either
  // voxels or hexahedra.  Eventually, we want to support all other cell
  // types as well (although, technically that ceases to be marching
  // cubes and becomes marching tetrahedra or the like).
  DAX_EXEC_EXPORT
  dax::Id operator()(
      const dax::exec::CellField<dax::Scalar,dax::CellTagVoxel> &values) const
  {
    const int voxelClass = GetVoxelClassification(IsoValue,values);
    return dax::worklet::internal::marchingcubes::NumFaces[voxelClass];
  }
private:
  dax::Scalar IsoValue;
};


// -----------------------------------------------------------------------------
class MarchingCubesTopology : public dax::exec::WorkletInterpolatedCell
{
public:

  typedef void ControlSignature(Topology, Geometry(Out), Field(Point,In));
  typedef void ExecutionSignature(Vertices(_1), _2, _3, VisitIndex);

  DAX_CONT_EXPORT MarchingCubesTopology(dax::Scalar isoValue)
    : IsoValue(isoValue){ }


  // TODO: This method should be overloaded such that it accepts either
  // voxels or hexahedra.  Eventually, we want to support all other cell
  // types as well (although, technically that ceases to be marching
  // cubes and becomes marching tetrahedra or the like).
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<dax::CellTagVoxel>& verts,
      dax::exec::InterpolatedCellPoints<dax::CellTagTriangle>& outCell,
      const dax::exec::CellField<dax::Scalar,dax::CellTagVoxel> &values,
      dax::Id inputCellVisitIndex) const
  {
    using dax::worklet::internal::marchingcubes::TriTable;
    // These should probably be available through the voxel class
    const unsigned char voxelVertEdges[12][2] ={
        {0,1}, {1,2}, {3,2}, {0,3},
        {4,5}, {5,6}, {7,6}, {4,7},
        {0,4}, {1,5}, {2,6}, {3,7},
      };

    const int voxelClass = GetVoxelClassification(IsoValue, values);

    //save the point ids and ratio to interpolate the points of the new cell
    for (dax::Id outVertIndex = 0;
         outVertIndex < outCell.NUM_VERTICES;
         ++outVertIndex)
      {
      const unsigned char edge = TriTable[voxelClass][(inputCellVisitIndex*3)+outVertIndex];
      const int vertA = voxelVertEdges[edge][0];
      const int vertB = voxelVertEdges[edge][1];

      // Find the weight for linear interpolation
      const dax::Scalar weight = (IsoValue - values[vertA]) /
                                (values[vertB]-values[vertA]);

      outCell.SetInterpolationPoint(outVertIndex,
                                    verts[vertA],
                                    verts[vertB],
                                    weight);
      }
  }

private:
  dax::Scalar IsoValue;
};
}
} //dax::worklet

#endif
