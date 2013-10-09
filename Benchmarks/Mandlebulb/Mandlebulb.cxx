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


#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/opengl/TransferToOpenGL.h>


//compute the mandlebulbs values for each point
mandle::MandlebulbVolume computeMandlebulb( dax::Vector3 origin,
                                          dax::Vector3 spacing,
                                          dax::Extent3 extent)
{
  //construct the dataset from -2,-2,-2, to 2,2,2
  //with the given spacing
  mandle::MandlebulbVolume vol(origin,spacing,extent);

  //compute the escape iterations for each point in the grid
  dax::cont::Scheduler< > scheduler;
  scheduler.Invoke( worklet::Mandlebulb(), vol.Grid.GetPointCoordinates(),
                    vol.EscapeIteration );

  vol.EscapeIteration.GetPortalConstControl();

  return vol;
}

//compute the surface of the mandlebulb for a given iteration
mandle::MandlebulbSurface extractSurface( mandle::MandlebulbVolume vol,
                                          dax::Scalar iteration )
{
  //lets extract the surface where the iteration value is greater than
  //the passed in iteration value
  mandle::MandlebulbSurface surface; //construct surface struct
  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Scheduler< > scheduler;

  //run the classify step
  dax::worklet::MarchingCubesClassify classify(iteration);
  scheduler.Invoke( classify,
                    vol.Grid,
                    vol.EscapeIteration,
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
                   vol.Grid,
                   surface.Data,
                   vol.EscapeIteration);

  // //generate a color for each point based on the escape iteration
  scheduler.Invoke(worklet::ColorsAndNorms(),
                    surface.Data,
                    surface.Data.GetPointCoordinates(),
                    surface.Colors,
                    surface.Norms);

  vol.EscapeIteration.ReleaseResourcesExecution();
  return surface;
}

//compute the slice of the volume for a given iteration
//sneaky we are going to modify iteration in this method on purpose
mandle::MandlebulbSurface extractSlice( mandle::MandlebulbVolume vol,
                                        dax::Scalar& iteration )
{
  dax::Vector3 origin = vol.Grid.GetOrigin();
  dax::Vector3 spacing = vol.Grid.GetSpacing();
  dax::Id3 dims = dax::extentCellDimensions(vol.Grid.GetExtent());

  //slicing at the edges where nothing is causes problems
  dax::Id index = dax::math::Max((int)iteration,5);
  index = dax::math::Min(index,dax::Id(20));
  iteration = index;

  dax::Vector3 location(origin[0] + spacing[0] * (index * (dims[0]/30) ),
                        origin[1] + spacing[1] * (dims[1]/2),
                        origin[2] + spacing[2] * (dims[2]/2) );

  dax::Vector3 normal(1,0,0);

  //lets extract the slice
  mandle::MandlebulbSurface surface;

  dax::cont::ArrayHandle<dax::Id> classification;

  dax::cont::Scheduler< > scheduler;

  //run the classify step
  dax::worklet::SliceClassify classify(location, normal);
  scheduler.Invoke( classify,
                    vol.Grid,
                    vol.Grid.GetPointCoordinates(),
                    classification );

  //setup the info for the second step
  dax::worklet::SliceGenerate generateSurface(location, normal);
  dax::cont::GenerateInterpolatedCells<
      dax::worklet::SliceGenerate > genWrapper(classification,generateSurface);

  //remove duplicates on purpose
  genWrapper.SetRemoveDuplicatePoints(false);

  //run the second step
  scheduler.Invoke(genWrapper,
                   vol.Grid,
                   surface.Data,
                   vol.Grid.GetPointCoordinates());

  // //generate a color for each point based on the escape iteration
  scheduler.Invoke(worklet::ColorsAndNorms(),
                    surface.Data,
                    surface.Data.GetPointCoordinates(),
                    surface.Colors,
                    surface.Norms);

  vol.EscapeIteration.ReleaseResourcesExecution();

  return surface;
}

void bindSurface( mandle::MandlebulbSurface& surface,
                  GLuint& coord,
                  GLuint& color,
                  GLuint& norm )
{
  //TransferToOpenGL will do the binding to the given buffers if needed
  dax::opengl::TransferToOpenGL(surface.Data.GetPointCoordinates(), coord);
  dax::opengl::TransferToOpenGL(surface.Colors, color);
  dax::opengl::TransferToOpenGL(surface.Norms, norm);

  //no need to keep the cuda side, as the next re-computation will have
  //redo all the work for all three of these
  surface.Data.GetPointCoordinates().GetPortalConstControl();
  surface.Data.GetPointCoordinates().ReleaseResourcesExecution();
  surface.Colors.ReleaseResourcesExecution();
  surface.Norms.ReleaseResourcesExecution();

}
