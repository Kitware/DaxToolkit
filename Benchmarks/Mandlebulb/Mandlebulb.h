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
#include <dax/cont/Scheduler.h>
#include <dax/math/Exp.h>
#include <dax/math/Trig.h>
#include <dax/opengl/TransferToOpenGL.h>
#include <dax/worklet/MarchingCubes.h>
#include <dax/worklet/Slice.h>

#include "CoolWarmColorMap.h"

//helper structs to make it easier to pass data around between functions
namespace mandle {

  struct MandlebulbInfo
  {
  MandlebulbInfo(  )
  {
    Grid.SetOrigin( dax::make_Vector3(-1,-1,-1) );
    Grid.SetExtent( dax::make_Id3(0,0,0),
                    dax::make_Id3(350,350,200) );
    Grid.SetSpacing( dax::make_Vector3(0.01,0.01,0.02) );
  }

  dax::cont::UniformGrid< > Grid;
  dax::cont::ArrayHandle<dax::Scalar> EscapeIteration;
  };

  struct MandlebulbSurface
  {
  typedef dax::cont::UnstructuredGrid< dax::CellTagTriangle > DataType;
  DataType Data;

  //dax doesn't currently have a per vertex worklet type
  //so for use to generate per vertex colors we have to manually make
  //types that represent the vertex values per cell

  //colors for all three verts of a triangle
  typedef dax::Tuple<unsigned char,12> ColorType;
  typedef dax::Tuple<dax::Scalar,9> NormType;

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
  for (dax::Id i=0; i < 32; ++i)
    {
    const dax::Scalar r = dax::math::Sqrt( dax::dot(pos,pos) );
    const dax::Scalar t = dax::math::Sqrt(pos[0] * pos[0] + pos[1] * pos[1]);
    const dax::Scalar theta = 8 * dax::math::ATan2( t, pos[2]);
    const dax::Scalar phi = 8 * dax::math::ATan2( pos[1], pos[2] );

    const dax::Scalar powR = dax::math::Pow(r,8);
    pos[0] = powR * dax::math::Sin(theta) * dax::math::Cos(phi) + inCoordinate[0];
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

using namespace mandle;

//compute the mandlebulbs values for each point
MandlebulbInfo computeMandlebulb(  )
{
  //construct the dataset from -2,-2,-2, to 2,2,2
  //with the given spacing
  MandlebulbInfo info;

  //compute the escape iterations for each point in the grid
  dax::cont::Scheduler< > scheduler;
  scheduler.Invoke( worklet::Mandlebulb(), info.Grid.GetPointCoordinates(),
                    info.EscapeIteration );

  return info;
}

//compute the surface of the mandlebulb for a given iteration
MandlebulbSurface extractSurface(  MandlebulbInfo info, dax::Scalar iteration )
{
  //lets extract the surface where the iteration value is greater than
  //the passed in iteration value
  MandlebulbSurface surface; //construct surface struct
  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Scheduler< > scheduler;

  //run the classify step
  dax::worklet::MarchingCubesClassify classify(iteration);
  scheduler.Invoke( classify,
                    info.Grid,
                    info.EscapeIteration,
                    classification );

  //setup the info for the second step
  dax::worklet::MarchingCubesGenerate generateSurface(iteration);
  dax::cont::GenerateInterpolatedCells<
      dax::worklet::MarchingCubesGenerate > genWrapper(classification,
                                                       generateSurface);

  //remove duplicates on purpose
  genWrapper.SetRemoveDuplicatePoints(false);

  //run the second step
  scheduler.Invoke(genWrapper,
                   info.Grid,
                   surface.Data,
                   info.EscapeIteration);

  // //generate a color for each point based on the escape iteration
  scheduler.Invoke(worklet::ColorsAndNorms(),
                    surface.Data,
                    surface.Data.GetPointCoordinates(),
                    surface.Colors,
                    surface.Norms);
  return surface;
}

//compute the slice of the volume for a given iteration
//sneaky we are going to modify iteration in this method on purpose
MandlebulbSurface extractSlice(  MandlebulbInfo info, dax::Scalar& iteration )
{
  dax::Vector3 origin = info.Grid.GetOrigin();
  dax::Vector3 spacing = info.Grid.GetSpacing();
  dax::Id3 dims = dax::extentCellDimensions(info.Grid.GetExtent());

  //slicing at the edges where nothing is causes problems
  dax::Id index = dax::math::Max((int)iteration,5);
  index = dax::math::Min(index,dax::Id(20));
  iteration = index;

  dax::Vector3 location(origin[0] + spacing[0] * (index * (dims[0]/30) ),
                        origin[1] + spacing[1] * (dims[1]/2),
                        origin[2] + spacing[2] * (dims[2]/2) );

  dax::Vector3 normal(1,0,0);

  //lets extract the slice
  MandlebulbSurface surface;

  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Scheduler< > scheduler;

  //run the classify step
  dax::worklet::SliceClassify classify(location, normal);
  scheduler.Invoke( classify,
                    info.Grid,
                    info.Grid.GetPointCoordinates(),
                    classification );

  //setup the info for the second step
  dax::worklet::SliceGenerate generateSurface(location, normal);
  dax::cont::GenerateInterpolatedCells<
      dax::worklet::SliceGenerate > genWrapper(classification,generateSurface);

  //remove duplicates on purpose
  genWrapper.SetRemoveDuplicatePoints(false);

  //run the second step
  scheduler.Invoke(genWrapper,
                   info.Grid,
                   surface.Data,
                   info.Grid.GetPointCoordinates());

  // //generate a color for each point based on the escape iteration
  scheduler.Invoke(worklet::ColorsAndNorms(),
                    surface.Data,
                    surface.Data.GetPointCoordinates(),
                    surface.Colors,
                    surface.Norms);
  return surface;
}

void bindSurface(MandlebulbSurface& surface, GLuint& coord,
                 GLuint& color, GLuint& norm )
{
  //TransferToOpenGL will do the binding to the given buffers if needed
  dax::opengl::TransferToOpenGL(surface.Data.GetPointCoordinates(), coord);
  dax::opengl::TransferToOpenGL(surface.Colors, color);
  dax::opengl::TransferToOpenGL(surface.Norms, norm);
}

#endif
