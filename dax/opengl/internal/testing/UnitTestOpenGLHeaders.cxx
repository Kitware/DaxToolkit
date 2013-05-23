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
#include <dax/opengl/internal/OpenGLHeaders.h>
#include <dax/cont/testing/Testing.h>

namespace {
void TestOpenGLHeaders()
{
#if defined(GL_VERSION_1_3) && (GL_VERSION_1_3 == 1)
  //this is pretty simple, we just verify that certain symbols exist
  //and the version of openGL is high enough that our interop will work.
  GLenum e = GL_ELEMENT_ARRAY_BUFFER;
  GLuint u = 1;
  u = u * e;
#else
  unable_to_find_required_gl_version();
#endif
}

}

int UnitTestOpenGLHeaders(int, char *[])
{
 return dax::cont::testing::Testing::Run(TestOpenGLHeaders);
}
