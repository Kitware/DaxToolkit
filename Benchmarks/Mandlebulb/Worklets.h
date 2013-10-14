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
#include <dax/cont/arg/ExecutionObject.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/math/Compare.h>
#include <dax/math/Exp.h>
#include <dax/math/Trig.h>
#include <dax/math/VectorAnalysis.h>

#include <dax/exec/WorkletMapField.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/worklet/MarchingCubes.h>


#include "CoolWarmColorMap.h"


namespace worklet {


//basic implementation of computing the Mandlebulb if a point
//escapes the mandlebulb of the 8th order
class Mandlebulb : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In), Field(Out));
  typedef _2 ExecutionSignature(_1);

  DAX_EXEC_EXPORT
  dax::Scalar operator()(dax::Vector3 inCoordinate) const
  {
  //going to compute the 8th order mandlebulb. This step computes
  //the iteration that a given point escapes at
  dax::Vector3 pos(0.0);

  //find the iteration we escape on
  for (dax::Id i=0; i < 35; ++i)
    {
    const dax::Scalar r = dax::math::Sqrt( dax::dot(pos,pos) );
    const dax::Scalar t = dax::math::Sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
    const dax::Scalar theta = 10 * dax::math::ATan2( t, pos[2]);
    const dax::Scalar phi = 10 * dax::math::ATan2( pos[1], pos[2] );

    const dax::Scalar powR = dax::math::Pow(r,10);
    pos[0] = powR * dax::math::Sin(theta) * dax::math::Tan(phi) + inCoordinate[0];
    pos[1] = powR * dax::math::Sin(theta) * dax::math::Sin(phi) + inCoordinate[1];
    pos[2] = powR * dax::math::Cos(theta) + inCoordinate[2];
    if(dax::dot(pos,pos) > 2)
      {
      return i;
      }
    }
  return 0;
  }
};

//does a combined clip and marching cubes at the same time
class MandlebulbClipClassify : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(Point), Field(Point), Field(Out));
  typedef _4 ExecutionSignature(_2, _3);

  DAX_CONT_EXPORT MandlebulbClipClassify(dax::Vector3 origin,
                                        dax::Vector3 location,
                                        dax::Vector3 normal,
                                        dax::Scalar isoValue)
  : Origin(origin),
    Location(location),
    Normal(normal),
    MClassify(isoValue)
  {
  }

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Scalar operator()(
    const dax::exec::CellField<dax::Vector3,CellTag> &coords,
    const dax::exec::CellField<dax::Scalar,CellTag> &values) const
  {
    dax::Id faces = 0;
    const bool is_good = this->InClipArea(coords,
                   typename dax::CellTraits<CellTag>::CanonicalCellTag());
    if(is_good)
      {
      //if we intersect the slice plane generate faces
      faces = this->MClassify(values);
      }
    return faces;
  }
 private:

  template<class CellTag>
    DAX_EXEC_EXPORT
    bool InClipArea(
      const dax::exec::CellField<dax::Vector3,CellTag> &coords,
      dax::CellTagHexahedron) const
    {
      //compute the location
      const dax::Scalar loca_value = dax::dot(Normal,Location);
      const int voxelClass =(
            ( dax::dot(Normal, coords[0] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[1] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[2] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[3] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[4] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[5] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[6] - Origin ) > loca_value ) |
            ( dax::dot(Normal, coords[7] - Origin ) > loca_value ) );
      return voxelClass != 0;
    }

    dax::Vector3 Origin;
    dax::Vector3 Location;
    dax::Vector3 Normal;
    dax::worklet::MarchingCubesClassify MClassify;
};


//basic implementation of computing color and norms for triangles
//since dax doesn't have a per vert worklet we are going to replicate
//that worklet type by using worklet map field
class ColorsAndNorms : public dax::exec::WorkletMapField
{
public:
  typedef void ControlSignature(Field(In),ExecObject(), Field(Out), Field(Out));
  typedef void ExecutionSignature(_1, _2, _3, _4);

  DAX_EXEC_EXPORT void operator()( dax::Id vert_index,
                                   const mandle::SurfaceCoords& coords,
                                   dax::Vector3& norm,
                                   dax::Tuple<unsigned char,4>& color  ) const
  {
  //compute normals
  const dax::Id cell_index = 3 * (vert_index / 3);
  const dax::Vector3 first_coord = coords[cell_index];

  const  dax::Vector3 a = coords[cell_index+1] - first_coord;
  const  dax::Vector3 b = coords[cell_index+2] - first_coord;
  norm = dax::math::Normal( dax::math::Cross(a,b) );

  //compute color field, wrap around in both directions
  //with an expanding color field from zero to 1.0
  const dax::Scalar s = dax::math::Abs( dax::dot(coords[vert_index],
                                          dax::make_Vector3(0.09,0.09,0.9)));
  const mandle::CoolWarmColorMap::ColorType &c = this->ColorMap.GetColor(s);
  color[0] = c[0];
  color[1] = c[1];
  color[2] = c[2];
  color[3] = 255;
  }

private:
  mandle::CoolWarmColorMap ColorMap;
};

}