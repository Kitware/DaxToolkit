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

#ifndef __PipelinePISTON_h
#define __PipelinePISTON_h

#include <dax/cont/UniformGrid.h>

class vtkImageData;

void RunPipelinePISTON(int pipeline, const dax::cont::UniformGrid &dgrid, vtkImageData* grid);

#endif //__PipelinePISTON_h
