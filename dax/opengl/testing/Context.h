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
#ifndef __dax__opengl__testing__Context_h
#define __dax__opengl__testing__Context_h

//constructs a valid openGL context so that we can verify
//that dax to open gl bindings work

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__)
# include <GLUT/glut.h>
#else
# include <GL/freeglut.h>
#endif

//#include glew before we include any other opengl headers
#include <dax/opengl/internal/OpenGLHeaders.h>
#include <dax/cont/ErrorControlBadValue.h>

#ifdef DAX_CUDA
# include <dax/cuda/cont/ChooseCudaDevice.h>
# include <dax/cuda/opengl/SetOpenGLDevice.h>
#endif

namespace dax{
namespace opengl {
namespace testing {

class Context
{
public:
  Context()
  {
  //lets create a window and context
    int argc=0;
    char** argv = 0;
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowPosition(0,0);
    glutInitWindowSize(512,512);
    glutCreateWindow("Testing Context");

    glewInit();
    if(!glewIsSupported("GL_VERSION_2_0"))
      {
      throw dax::cont::ErrorControlBadValue(
                                  "Unable to create an OpenGL 2.0 Context");
      }

#ifdef DAX_CUDA
    int id = dax::cuda::cont::FindFastestDeviceId();
    dax::cuda::opengl::SetCudaGLDevice(id);
#endif
  }
};

}
}
}
#endif

