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

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/exec/ExecutionObjectBase.h>

#include <dax/opengl/internal/OpenGLHeaders.h>

//helper structs to make it easier to pass data around between functions
namespace mandle
{

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

  typedef dax::Tuple<unsigned char,4> ColorType;


  public:

  DataType Data;
  dax::cont::ArrayHandle<ColorType> Colors;
  dax::cont::ArrayHandle<dax::Vector3> Norms;
  };

  class SurfaceCoords : public dax::exec::ExecutionObjectBase
  {
    typedef dax::cont::UnstructuredGrid< dax::CellTagTriangle > DataType;
    typedef DataType::PointCoordinatesType CoordType;
    typedef CoordType::PortalConstExecution PortalType;

  public:
    DAX_CONT_EXPORT
    SurfaceCoords( DataType grid ):
      Coords( grid.GetPointCoordinates().PrepareForInput() )
      {
      }

  DAX_EXEC_EXPORT dax::Vector3 operator[](int idx) const {
      return this->Coords.Get(idx);
      }

  private:
    PortalType Coords;
  };

}

//define functions to compute the mandlebulb info
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                            dax::Vector3 spacing,
                                            dax::Extent3 extent);

mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume vol,
                                          dax::Scalar iteration );

//cut percent represents the ratio from 0 - 1 that we want the cut
//to be along the axis
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume vol,
                                        dax::Scalar cut_percent,
                                        dax::Scalar iteration );

void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm );

#endif
