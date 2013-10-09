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
#ifndef __dax__benchmarks_Mandlebulb_h
#define __dax__benchmarks_Mandlebulb_h

#include <iostream>

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/Extent.h>
#include <dax/math/Compare.h>
#include <dax/math/Exp.h>
#include <dax/math/Trig.h>
#include <dax/math/VectorAnalysis.h>
#include <dax/opengl/internal/OpenGLHeaders.h>

#include <dax/exec/WorkletMapField.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/worklet/MarchingCubes.h>

#include "CoolWarmColorMap.h"

//helper structs to make it easier to pass data around between functions
namespace mandle {

  class MandlebulbVolume
  {
  public:
  MandlebulbVolume() { }

  MandlebulbVolume( dax::Vector3 origin,
                   dax::Vector3 spacing,
                   dax::Extent3 extent )
  {
    Grid.SetOrigin ( origin  );
    Grid.SetExtent ( extent  );
    Grid.SetSpacing( spacing );
  }

  dax::cont::UniformGrid< > Grid;
  dax::cont::ArrayHandle<dax::Scalar> EscapeIteration;
  };

  class MandlebulbSurface
  {
  typedef dax::cont::UnstructuredGrid< dax::CellTagTriangle > DataType;

  //dax doesn't currently have a per vertex worklet type
  //so for use to generate per vertex colors we have to manually make
  //types that represent the vertex values per cell

  //colors for all three verts of a triangle
  typedef dax::Tuple<unsigned char,12> ColorType;
  typedef dax::Tuple<dax::Scalar,9> NormType;

  public:

  DataType Data;
  dax::cont::ArrayHandle<ColorType> Colors;
  dax::cont::ArrayHandle<NormType> Norms;
  };
}

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

//does a combined slice and marching cubes at the same time
class MandlebulbCutClassify : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(Point), Field(Point), Field(Out));
  typedef _4 ExecutionSignature(_2, _3);

  DAX_CONT_EXPORT MandlebulbCutClassify(dax::Vector3 origin,
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
    const bool is_good = this->InCutArea(coords,
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
    bool InCutArea(
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
class ColorsAndNorms : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(In,Point), Field(Out), Field(Out));
  typedef void ExecutionSignature(_2, _3, _4);

  DAX_EXEC_EXPORT
  void operator()(
        dax::exec::CellField<dax::Vector3, dax::CellTagTriangle> coordinates,
        dax::Tuple<unsigned char,12>& colors,
        dax::Tuple<dax::Scalar,9>& norms) const
  {
  //compute normals
  const  dax::Vector3 a = coordinates[1] - coordinates[0];
  const  dax::Vector3 b = coordinates[2] - coordinates[0];
         dax::Vector3 n = dax::math::Normal( dax::math::Cross(a,b) );

  norms[6] = norms[3] = norms[0] = n[0];
  norms[7] = norms[4] = norms[1] = n[1];
  norms[8] = norms[5] = norms[2] = n[2];

  //compute color field, wrap around in both directions
  //with an expanding color field from zero to 1.0
  for (int i=0; i < 3; ++i)
    {
    const dax::Scalar s =
        dax::math::Abs(dax::dot(coordinates[i],
                                dax::make_Vector3(0.09,0.09,0.9)));
    const mandle::CoolWarmColorMap::ColorType &c = this->ColorMap.GetColor(s);
    colors[i*4+0] = c[0];
    colors[i*4+1] = c[1];
    colors[i*4+2] = c[2];
    colors[i*4+3] = 255;
    }
  }

private:
  mandle::CoolWarmColorMap ColorMap;
};

}

//define functions to compute the mandlebulb info
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                            dax::Vector3 spacing,
                                            dax::Extent3 extent);

mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume vol,
                                          dax::Scalar iteration );

//slice percent represents the ratio from 0 - 1 that we want the slice
//to be along the axis
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume vol,
                                        dax::Scalar cut_percent,
                                        dax::Scalar iteration );

void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm );

#endif
