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

#ifndef __dax__opengl__internal_TransferToOpenGL_h
#define __dax__opengl__internal_TransferToOpenGL_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayContainerControlBasic.h>

#include <dax/opengl/internal/OpenGLHeaders.h>
#include <dax/opengl/internal/BufferTypePicker.h>

namespace dax {
namespace opengl {
namespace internal {

namespace detail
{
template< class ContainerTag, class ValueType, class DeviceAdapterTag >
DAX_CONT_EXPORT
void CopyFromHandle( ContainerTag,
  dax::cont::ArrayHandle<ValueType, ContainerTag, DeviceAdapterTag>& handle,
  GLenum type)
{
  //Generic implementation that will work no matter what. We copy
  //back down to a local temporary array and give that to OpenGL to upload
  //to the rendering system
  const dax::Id numberOfValues = handle.GetNumberOfValues();
  std::size_t size = sizeof(ValueType) * numberOfValues;

  ValueType* temporaryStorage = new ValueType[numberOfValues];

  //Detach the current buffer
  glBufferData(type, size, 0, GL_DYNAMIC_DRAW);

  handle.CopyInto(temporaryStorage);
  glBufferSubData(type,0,size,temporaryStorage);

  delete[] temporaryStorage;
}

template< class ValueType, class DeviceAdapterTag >
DAX_CONT_EXPORT
void CopyFromHandle( dax::cont::ArrayContainerControlTagBasic,
  dax::cont::ArrayHandle<ValueType,
    dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>& handle,
  GLenum type)
{
  //Specialization given that we are use an C allocated array container tag
  //that allows us to directly hook in and use the execution portals
  //iterators as memory references. This also works because we know
  //that this class isn't used for cuda interop, instead we are specialized
  //in that case
  typedef dax::cont::ArrayHandle<ValueType,
    dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag> HandleType;
  typedef typename HandleType::PortalConstExecution PortalType;

  const dax::Id numberOfValues = handle.GetNumberOfValues();
  std::size_t size = sizeof(ValueType) * numberOfValues;

  //Detach the current buffer
  glBufferData(type, size, 0, GL_DYNAMIC_DRAW);

  //Allocate the memory and set it as static draw and copy into opengl
  PortalType portal = handle.PrepareForInput();
  glBufferSubData(type,0,size,portal.GetIteratorBegin());
}

} //namespace detail

/// \brief Manages transferring an ArrayHandle to opengl .
///
/// \c TransferToOpenGL manages to transfer the contents of an ArrayHandle
/// to OpenGL as efficiently as possible.
///
template<typename ValueType, class DeviceAdapterTag>
class TransferToOpenGL
{
public:
  DAX_CONT_EXPORT TransferToOpenGL():
  Type( dax::opengl::internal::BufferTypePicker( ValueType() ) )
  {}

  DAX_CONT_EXPORT explicit TransferToOpenGL(GLenum type):
  Type(type)
  {}

  DAX_CONT_EXPORT GLenum GetType() const { return this->Type; }

  template< typename ContainerTag >
  DAX_CONT_EXPORT
  void Transfer (
    dax::cont::ArrayHandle<ValueType, ContainerTag, DeviceAdapterTag>& handle,
    GLuint& openGLHandle ) const
  {
  //make a buffer for the handle if the user has forgotten too
  if(!glIsBuffer(openGLHandle))
    {
    glGenBuffers(1,&openGLHandle);
    }

  //bind the buffer to the given buffer type
  glBindBuffer(this->Type, openGLHandle);

  //transfer the data.
  //the primary concern that we have at this point is data locality and
  //the type of container. Our options include using CopyInto and provide a
  //temporary location for the values to reside before we give it to openGL
  //this works for all container types.
  //
  //second option is to call PrepareForInput and get a PortalConstExecution.
  //if we are BasicContainerType this would allow us the ability to grab
  //the raw memory value and copy those, which we know are valid and remove
  //a unneeded copy.
  //
  //The end result is that we have CopyFromHandle which does number two
  //from ArrayContainerControlBasic, and does the CopyInto for everything else
  detail::CopyFromHandle(ContainerTag(), handle, this->Type);
  }
private:
  GLenum Type;
};

}
}
} //namespace dax::opengl::internal

//-----------------------------------------------------------------------------
// These includes are intentionally placed here after the declaration of the
// TransferToOpenGL class, so that people get the correct device adapter
/// specializations if they exist.
#if DAX_DEVICE_ADAPTER == DAX_DEVICE_ADAPTER_CUDA
#include <dax/cuda/opengl/internal/TransferToOpenGL.h>
#endif


#endif
