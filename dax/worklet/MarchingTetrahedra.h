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
//  Copyright 2014 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#ifndef __MarchingTetrahedra_worklet_
#define __MarchingTetrahedra_worklet_

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/InterpolatedCellPoints.h>
#include <dax/exec/WorkletInterpolatedCell.h>
#include <dax/exec/WorkletMapCell.h>

#include <dax/worklet/internal/MarchingTetrahedraTable.h>

namespace dax{
namespace worklet{

namespace internal{
namespace marchingtetrahedra{
//------------------------------------------------------------------------------
template<typename T, typename U>
DAX_EXEC_EXPORT
int GetTetrahedronClassification(const T isoValue, const U& values)
{
    return ((values[0] > isoValue) << 0 |
            (values[1] > isoValue) << 1 |
            (values[2] > isoValue) << 2 |
            (values[3] > isoValue) << 3 );
}
}
}

//------------------------------------------------------------------------------
class MarchingTetrahedraCount : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(TopologyIn, FieldPointIn, FieldOut);
  typedef _3 ExecutionSignature(_2);

  DAX_CONT_EXPORT MarchingTetrahedraCount(dax::Scalar isoValue)
      : IsoValue(isoValue) {  }

  template<typename T, class CellTag>
  DAX_EXEC_EXPORT
  dax::Id operator()(
          const dax::exec::CellField<T, CellTag> &values) const
  {
      // If you get a compile error on the following line, it means that this
      // worklet was used with an improper cell type.  Check the cell type for the
      // input grid given in the control environment.
      return this->GetNumFaces(
            values,
             typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }
private:
  dax::Scalar IsoValue;

  template<typename T, class CellTag>
  DAX_EXEC_EXPORT
  dax::Id GetNumFaces(const dax::exec::CellField<T,CellTag> &values,
                      dax::CellTagTetrahedron) const
  {
    const int voxelClass =
    internal::marchingtetrahedra::GetTetrahedronClassification(IsoValue,values);
    return dax::worklet::internal::marchingtetrahedra::NumFaces[voxelClass];
  }
};

//------------------------------------------------------------------------------
class MarchingTetrahedraGenerate : public dax::exec::WorkletInterpolatedCell
{
public:

  typedef void ControlSignature(TopologyIn, GeometryOut, FieldPointIn);
  typedef void ExecutionSignature(AsVertices(_1), _2, _3, VisitIndex);

  DAX_CONT_EXPORT MarchingTetrahedraGenerate(dax::Scalar isoValue)
        : IsoValue(isoValue) {   }

  template<class CellTag>
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<CellTag>& verts,
      dax::exec::InterpolatedCellPoints<dax::CellTagTriangle>& outCell,
      const dax::exec::CellField<dax::Scalar,CellTag> &values,
      dax::Id inputCellVisitIndex) const
  {
    // If you get a compile error on the following line, it means that this
    // worklet was used with an improper cell type.  Check the cell type for the
    // input grid given in the control environment.
    this->BuildTriangle(
          verts,
          outCell,
          values,
          inputCellVisitIndex,
          typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }

private:
  dax::Scalar IsoValue;

  template<class CellTag>
  DAX_EXEC_EXPORT void BuildTriangle(
      const dax::exec::CellVertices<CellTag>& verts,
      dax::exec::InterpolatedCellPoints<dax::CellTagTriangle>& outCell,
      const dax::exec::CellField<dax::Scalar,CellTag> &values,
      dax::Id inputCellVisitIndex,
      dax::CellTagTetrahedron) const
  {
    using dax::worklet::internal::marchingtetrahedra::TriTable;
    // These should probably be available through the voxel class
    const unsigned char cellVertEdges[6][2] ={
        {0,1}, {1,2}, {2,0},
        {3,0}, {3,1}, {3,2}
      };

    const int voxelClass =
        internal::marchingtetrahedra::GetTetrahedronClassification(IsoValue,values);

    //save the point ids and ratio to interpolate the points of the new cell
    for (dax::Id outVertIndex = 0;
         outVertIndex < outCell.NUM_VERTICES;
         ++outVertIndex)
      {
      const unsigned char edge = TriTable[voxelClass][(inputCellVisitIndex*3)+outVertIndex];
      int vertA;
      int vertB;
      if (verts[cellVertEdges[edge][0]] < verts[cellVertEdges[edge][1]])
        {
        vertA = cellVertEdges[edge][0];
        vertB = cellVertEdges[edge][1];
        }
      else
        {
        vertA = cellVertEdges[edge][1];
        vertB = cellVertEdges[edge][0];
        }

      // Find the weight for linear interpolation
      const dax::Scalar weight = (IsoValue - values[vertA]) /
                                (values[vertB]-values[vertA]);

      outCell.SetInterpolationPoint(outVertIndex,
                                    verts[vertA],
                                    verts[vertB],
                                    weight);
      }
  }
};
}
}
#endif // __MarchingTetrahedra_worklet_
