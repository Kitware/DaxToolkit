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


//This sets up testing with the default device adapter and array container
#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/opengl/testing/TestingOpenGLInterop.h>

int UnitTestTransferToOpenGL(int, char *[])
{
  return dax::opengl::testing::TestingOpenGLInterop<
            dax::cont::DeviceAdapterTagSerial >::Run();
}
