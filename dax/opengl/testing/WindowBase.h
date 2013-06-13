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
#ifndef __dax__opengl__testing__WindowBase_h
#define __dax__opengl__testing__WindowBase_h

//constructs a valid openGL context so that we can verify
//that dax to open gl bindings work
#include <string>

// OpenGL Graphics includes
//glew needs to go before glut
#include <dax/opengl/internal/OpenGLHeaders.h>
#if defined (__APPLE__)
# include <GLUT/glut.h>
#else
# include <GL/glut.h>
#endif

#include <dax/cont/ErrorControlBadValue.h>

#ifdef DAX_CUDA
# include <dax/cuda/cont/ChooseCudaDevice.h>
# include <dax/cuda/opengl/SetOpenGLDevice.h>
#endif



namespace dax{
namespace opengl{
namespace testing{

namespace internal
{
template <typename T>
struct GLUTStaticCallbackHolder
{ static T* StaticGLUTResource; };

template <typename T>
T* GLUTStaticCallbackHolder<T>::StaticGLUTResource;

}


/// \brief Basic GLUT Wrapper class
///
/// This class gives the ability to wrap the glut function callbacks into
/// a single class so that you can use c++ objects. The only downside
/// is that you can only have a single window created
///
template< class Derived >
class WindowBase : private internal::GLUTStaticCallbackHolder<Derived>
{

public:
  void Init(std::string title, int width, int height,
            int argc, char** argv)
  {
  //set our selves as the static instance to call
  WindowBase<Derived>::StaticGLUTResource = static_cast<Derived*>(this);

  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE );
  glutInitWindowPosition(0,0);
  glutInitWindowSize(width,height);
  glutCreateWindow(title.c_str());

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

  //attach all the glut call backs
  glutDisplayFunc( WindowBase<Derived>::GLUTDisplayCallback );
  glutIdleFunc( WindowBase<Derived>::GLUTIdleCallback );
  glutReshapeFunc( WindowBase<Derived>::GLUTChangeSizeCallback );
  glutKeyboardFunc( WindowBase<Derived>::GLUTKeyCallback );
  glutSpecialFunc( WindowBase<Derived>::GLUTSpecialKeyCallback );
  glutMouseFunc( WindowBase<Derived>::GLUTMouseCallback );
  glutMotionFunc( WindowBase<Derived>::GLUTMouseMoveCallback );

  //call any custom init code you want to have
  WindowBase<Derived>::StaticGLUTResource->PostInit();
  }

  void Init(std::string title, int width, int height)
  {
    int argc=0;
    char** argv = 0;
    Init(title,width,height,argc,argv);
  }

  //Init must be called before you call Start so that we have a valid
  //opengl context
  void Start()
  {
    glutMainLoop();
  }


  static void GLUTDisplayCallback()
    { WindowBase<Derived>::StaticGLUTResource->Display(); }

  static void GLUTIdleCallback()
    { WindowBase<Derived>::StaticGLUTResource->Idle(); }

  static void GLUTChangeSizeCallback(int width, int height)
    { WindowBase<Derived>::StaticGLUTResource->ChangeSize(width,height); }

  static void GLUTKeyCallback(unsigned char key, int x, int y)
    { WindowBase<Derived>::StaticGLUTResource->Key(key,x,y); }

  static void GLUTSpecialKeyCallback(int key, int x, int y)
    { WindowBase<Derived>::StaticGLUTResource->SpecialKey(key,x,y); }

  static void GLUTMouseCallback(int button, int state ,int x, int y)
    { WindowBase<Derived>::StaticGLUTResource->Mouse(button,state,x,y); }

  static void GLUTMouseMoveCallback(int x, int y)
    { WindowBase<Derived>::StaticGLUTResource->MouseMove(x,y); }
};


}
}
}
#endif
