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

#ifndef __dax__opengl__TransferToOpenGL_h
#define __dax__opengl__TransferToOpenGL_h

#include <dax/cont/ArrayHandle.h>
#include <dax/opengl/internal/TransferToOpenGL.h>

namespace dax{
namespace opengl {
/// \brief Manages transferring an ArrayHandle to opengl .
///
/// \c TransferToOpenGL manages to transfer the contents of an ArrayHandle
/// to OpenGL as efficiently as possible. Will return the type of array buffer
/// that we have bound the handle too. Will be GL_ELEMENT_ARRAY_BUFFER for
/// dax::Id, and GL_ARRAY_BUFFER for everything else.
///
/// This function keeps the buffer as the active buffer of the returned type.
///
/// This function will throw exceptions if the transfer wasn't possible
///
template<typename T, class ContainerTag, class DeviceAdapterTag>
DAX_CONT_EXPORT
GLenum TransferToOpenGL(dax::cont::ArrayHandle<T,
                                               ContainerTag,
                                               DeviceAdapterTag> handle,
                        GLuint& openGLHandle)
{
  dax::opengl::internal::TransferToOpenGL<T, DeviceAdapterTag> toGL;
  toGL.Transfer(handle,openGLHandle);
  return toGL.GetType();
}

/// \brief Manages transferring an ArrayHandle to opengl .
///
/// \c TransferToOpenGL manages to transfer the contents of an ArrayHandle
/// to OpenGL as efficiently as possible. Will use the given type as how
/// to bind the buffer.
///
/// This function keeps the buffer as the active buffer of the input type.
///
/// This function will throw exceptions if the transfer wasn't possible
///
template<typename T, class ContainerTag, class DeviceAdapterTag>
DAX_CONT_EXPORT
void TransferToOpenGL(dax::cont::ArrayHandle<T,
                                             ContainerTag,
                                             DeviceAdapterTag> handle,
                      GLuint& openGLHandle,
                      GLenum type)
{
  dax::opengl::internal::TransferToOpenGL<T, DeviceAdapterTag> toGL(type);
  toGL.Transfer(handle,openGLHandle);
}

}}

#endif
