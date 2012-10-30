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

#ifndef __PipelineTBB_h
#define __PipelineTBB_h

#include <dax/cont/UniformGrid.h>
#include <dax/tbb/cont/DeviceAdapterTBB.h>

void RunPipelineTBB(
    const dax::cont::UniformGrid<dax::tbb::cont::DeviceAdapterTagTBB> &grid);

#endif //__PipelineTBB_h
