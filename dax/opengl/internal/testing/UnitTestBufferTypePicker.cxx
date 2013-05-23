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

#include <dax/opengl/internal/BufferTypePicker.h>
#include <dax/cont/testing/Testing.h>

namespace
{
void TestBufferTypePicker()
{
  //just verify that certain types match
  GLenum type;
  typedef unsigned int daxUint;

  type = dax::opengl::internal::BufferTypePicker(dax::Id());
  DAX_TEST_ASSERT(type == GL_ELEMENT_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(int());
  DAX_TEST_ASSERT(type == GL_ELEMENT_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(daxUint());
  DAX_TEST_ASSERT(type == GL_ELEMENT_ARRAY_BUFFER, "Bad OpenGL Buffer Type");

  type = dax::opengl::internal::BufferTypePicker(dax::Vector4());
  DAX_TEST_ASSERT(type == GL_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(dax::Vector3());
  DAX_TEST_ASSERT(type == GL_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(dax::Scalar());
  DAX_TEST_ASSERT(type == GL_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(float());
  DAX_TEST_ASSERT(type == GL_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
  type = dax::opengl::internal::BufferTypePicker(double());
  DAX_TEST_ASSERT(type == GL_ARRAY_BUFFER, "Bad OpenGL Buffer Type");
}
}


int UnitTestBufferTypePicker(int, char *[])
{
 return dax::cont::testing::Testing::Run(TestBufferTypePicker);
}
