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

#include "Mandlebulb.h"
#include "Worklets.h"

#include <iostream>

#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>

#include <dax/opengl/TransferToOpenGL.h>

namespace detail
{

  mandle::MandlebulbSurface generateSurface( mandle::MandlebulbVolume& vol,
                            dax::Scalar iteration,
                            dax::cont::ArrayHandle<dax::Id> classification)

{
  //find the default device adapter
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG AdapterTag;

  //Make it easy to call the DeviceAdapter with the right tag
  typedef dax::cont::DeviceAdapterAlgorithm<AdapterTag> DeviceAdapter;

  mandle::MandlebulbSurface surface; //construct surface struct

  dax::cont::Timer<> timer;

  //setup the info for the second step
  dax::worklet::MarchingCubesGenerate generateSurface(iteration);
  dax::cont::DispatcherGenerateInterpolatedCells<
      ::dax::worklet::MarchingCubesGenerate > surfDispacther(classification,
                                                             generateSurface);

  surfDispacther.SetRemoveDuplicatePoints(false);

  //run the second step
  surfDispacther.Invoke( vol.Grid, surface.Data, vol.EscapeIteration);

  std::cout << "mc stage 2: " << timer.GetElapsedTime() << std::endl;

  //generate a color for each point based on the escape iteration
  if(surface.Data.GetNumberOfPoints() > 0)
    {
    mandle::SurfaceCoords surface_coords(surface.Data);
    dax::cont::DispatcherMapField<worklet::ColorsAndNorms> colorNormsDispatcher;
    colorNormsDispatcher.Invoke(
                  dax::cont::make_ArrayHandleCounting(dax::Id(0),
                                          surface.Data.GetNumberOfPoints()),
                  surface_coords,
                  surface.Norms,
                  surface.Colors);
    std::cout << "colors & norms: " << timer.GetElapsedTime() << std::endl;
    }

  return surface;
}

}


//compute the mandlebulbs values for each point
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                          dax::Vector3 spacing,
                                          dax::Extent3 extent)
{
  //construct the dataset from -2,-2,-2, to 2,2,2
  //with the given spacing
  mandle::MandlebulbVolume vol(origin,spacing,extent);

  //compute the escape iterations for each point in the grid
  dax::cont::DispatcherMapField< worklet::Mandlebulb >().Invoke(
                                              vol.Grid.GetPointCoordinates(),
                                              vol.EscapeIteration );

  return vol;
}

//compute the surface of the mandlebulb for a given iteration
mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume& vol,
                                          dax::Scalar iteration )
{
  //lets extract the surface where the iteration value is greater than
  //the passed in iteration value

  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Timer<> timer;
  //run the classify step
  dax::cont::DispatcherMapCell< ::dax::worklet::MarchingCubesClassify >
        classify( (::dax::worklet::MarchingCubesClassify(iteration)) );
  classify.Invoke(vol.Grid, vol.EscapeIteration, classification );


  std::cout << "mc stage 1: " << timer.GetElapsedTime() << std::endl;

  return detail::generateSurface(vol,iteration,classification);
}

//compute the clip of the volume for a given iteration
mandle::MandlebulbSurface extractCut( mandle::MandlebulbVolume& vol,
                                        dax::Scalar cut_percent,
                                        dax::Scalar iteration )
{

  dax::Vector3 origin = vol.Grid.GetOrigin();
  dax::Vector3 spacing = vol.Grid.GetSpacing();
  dax::Id3 dims = dax::extentCellDimensions(vol.Grid.GetExtent());

  //slicing at the edges where nothing is causes problems
  //we are doing z slice so we have to go from positive
  dax::Vector3 location(origin[0] + spacing[0] * dims[0],
                        origin[1] + spacing[1] * dims[1],
                        origin[2] + spacing[2] * (dims[2] * cut_percent) );
  dax::Vector3 normal(0,0,1);

  //lets extract the clip
  mandle::MandlebulbSurface surface;
  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Timer<> timer;

  //run the classify step
  dax::cont::DispatcherMapCell< ::worklet::MandlebulbClipClassify >
      classify( (::worklet::MandlebulbClipClassify( origin, location, normal, iteration)) );
  classify.Invoke(vol.Grid,
                  vol.Grid.GetPointCoordinates(),
                  vol.EscapeIteration,
                  classification );
  std::cout << "mc stage 1: "  << timer.GetElapsedTime() << std::endl;

  return detail::generateSurface(vol,iteration,classification);
}


void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm )
{
  if(surface.Data.GetNumberOfPoints() == 0)
    return;

  //TransferToOpenGL will do the binding to the given buffers if needed
  dax::opengl::TransferToOpenGL(surface.Data.GetPointCoordinates(), coord);
  dax::opengl::TransferToOpenGL(surface.Colors, color);
  dax::opengl::TransferToOpenGL(surface.Norms, norm);

  //no need to keep the cuda side, as the next re-computation will have
  //redo all the work for all three of these
  surface.Colors.ReleaseResourcesExecution();
  surface.Norms.ReleaseResourcesExecution();

}
