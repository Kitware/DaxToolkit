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
#ifndef __Slice_worklet_
#define __Slice_worklet_

#include <dax/worklet/MarchingCubes.h>

namespace dax {
namespace worklet {

// -----------------------------------------------------------------------------
//In the future we could extend this to be more general so we can accept
//any implicit function, that would help reduce the duplicate code between
//this class and marching cubes
class SliceCount : public dax::exec::WorkletMapCell
{

public:
  typedef void ControlSignature(Topology, Field(Point), Field(Out));
  typedef _3 ExecutionSignature(_2);

  DAX_CONT_EXPORT SliceCount(dax::Vector3 origin, dax::Vector3 normal)
    : Origin(origin),
      Normal(normal)
  {
  }

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id operator()(
      const dax::exec::CellField<dax::Vector3,CellTag> &coords) const
  {
    // If you get a compile error on the following line, it means that this
    // worklet was used with an improper cell type.  Check the cell type for the
    // input grid given in the control environment.
    return this->GetNumFaces(
          coords,
          typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }
private:
  dax::Vector3 Origin;
  dax::Vector3 Normal;

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id GetNumFaces(const dax::exec::CellField<dax::Vector3,CellTag> &coords,
                      dax::CellTagHexahedron) const
  {
    //rather than compute a new cell field of scalars and use that
    //as input to GetHexahedronClassification I am in lining the calculation
    const dax::Scalar isoValue = dax::dot(Normal,Origin);
    const int voxelClass =(
          ( dax::dot(Normal, coords[0] - Origin ) > isoValue ) << 0 |
          ( dax::dot(Normal, coords[1] - Origin ) > isoValue ) << 1 |
          ( dax::dot(Normal, coords[2] - Origin ) > isoValue ) << 2 |
          ( dax::dot(Normal, coords[3] - Origin ) > isoValue ) << 3 |
          ( dax::dot(Normal, coords[4] - Origin ) > isoValue ) << 4 |
          ( dax::dot(Normal, coords[5] - Origin ) > isoValue ) << 5 |
          ( dax::dot(Normal, coords[6] - Origin ) > isoValue ) << 6 |
          ( dax::dot(Normal, coords[7] - Origin ) > isoValue ) << 7);
    return dax::worklet::internal::marchingcubes::NumFaces[voxelClass];
  }
};

// -----------------------------------------------------------------------------
//In the future we could extend this to be more general so we can accept
//any implicit function, that would help reduce the duplicate code between
//this class and marching cubes
class SliceGenerate : public dax::exec::WorkletInterpolatedCell
{
public:

  typedef void ControlSignature(Topology, Geometry(Out), Field(Point,In));
  typedef void ExecutionSignature(Vertices(_1), _2, _3, VisitIndex);

  DAX_CONT_EXPORT SliceGenerate(dax::Vector3 origin, dax::Vector3 normal)
    : Origin(origin),
      Normal(normal)
  {
  }

  template<class CellTag>
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<CellTag>& verts,
      dax::exec::InterpolatedCellPoints<dax::CellTagTriangle>& outCell,
      const dax::exec::CellField<dax::Vector3,CellTag> &coords,
      dax::Id inputCellVisitIndex) const
  {
    // If you get a compile error on the following line, it means that this
    // worklet was used with an improper cell type.  Check the cell type for the
    // input grid given in the control environment.
    this->BuildTriangle(
          verts,
          outCell,
          coords,
          inputCellVisitIndex,
          typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }

private:
  dax::Vector3 Origin;
  dax::Vector3 Normal;

  template<class CellTag>
  DAX_EXEC_EXPORT void BuildTriangle(
      const dax::exec::CellVertices<CellTag>& verts,
      dax::exec::InterpolatedCellPoints<dax::CellTagTriangle>& outCell,
      const dax::exec::CellField<dax::Vector3,CellTag> &coords,
      dax::Id inputCellVisitIndex,
      dax::CellTagHexahedron) const
  {
    using dax::worklet::internal::marchingcubes::TriTable;

    //covert the coordinates into an iso field, this allows us to compute
    //only once the interpolated iso value for each coordinate for this cell
    //otherwise we would have to compute the interpolate iso value multiple
    //times
    const dax::Scalar isoValue = dax::dot(this->Normal,this->Origin);
    dax::exec::CellField<dax::Scalar, CellTag > iso_values;
    for(dax::Id index = 0;
        index < dax::CellTraits<CellTag>::NUM_VERTICES;
        ++index)
      {
      iso_values[index] = dax::dot(this->Normal,coords[index]-this->Origin);
      }

    const int voxelClass =
      internal::marchingcubes::GetHexahedronClassification(isoValue,iso_values);

    // These should probably be available through the voxel class
    const unsigned char voxelVertEdges[12][2] ={
        {0,1}, {1,2}, {3,2}, {0,3},
        {4,5}, {5,6}, {7,6}, {4,7},
        {0,4}, {1,5}, {2,6}, {3,7},
      };

    //save the point ids and ratio to interpolate the points of the new cell
    for (dax::Id outVertIndex = 0;
         outVertIndex < outCell.NUM_VERTICES;
         ++outVertIndex)
      {
      const unsigned char edge = TriTable[voxelClass][(inputCellVisitIndex*3)+outVertIndex];
      const int vertA = voxelVertEdges[edge][0];
      const int vertB = voxelVertEdges[edge][1];

      // Find the weight for linear interpolation
      const dax::Scalar weight = (isoValue - iso_values[vertA]) /
                                (iso_values[vertB]-iso_values[vertA]);

      outCell.SetInterpolationPoint(outVertIndex,
                                    verts[vertA],
                                    verts[vertB],
                                    weight);
      }
  }
};

}
}
#endif
