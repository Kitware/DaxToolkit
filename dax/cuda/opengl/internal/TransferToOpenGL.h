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

#ifndef __dax__cuda__opengl__internal_TransferToOpenGL_h
#define __dax__cuda__opengl__internal_TransferToOpenGL_h

#include <dax/cont/ErrorExecution.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

#include <dax/cuda/cont/internal/SetThrustForCuda.h>
#include <dax/cuda/cont/internal/DeviceAdapterTagCuda.h>

#include <dax/opengl/internal/TransferToOpenGL.h>

#include <dax/thrust/cont/internal/Copy.h>


namespace dax {
namespace opengl {
namespace internal {

/// \brief Manages transferring an ArrayHandle to opengl .
///
/// \c TransferToOpenGL manages to transfer the contents of an ArrayHandle
/// to OpenGL as efficiently as possible.
///
template<typename ValueType>
class TransferToOpenGL<ValueType, dax::cuda::cont::DeviceAdapterTagCuda>
{
  typedef dax::cuda::cont::DeviceAdapterTagCuda DeviceAdapterTag;
public:
  DAX_CONT_EXPORT TransferToOpenGL():
  Type( dax::opengl::internal::BufferTypePicker( ValueType() ) )
  {}

  DAX_CONT_EXPORT explicit TransferToOpenGL(GLenum type):
  Type(type)
  {}

  GLenum GetType() const { return this->Type; }

  template< typename ContainerTag >
  DAX_CONT_EXPORT
  void Transfer (
    dax::cont::ArrayHandle<ValueType, ContainerTag, DeviceAdapterTag>& handle,
    GLuint& openGLHandle ) const
  {
  //construct a cuda resource handle
  cudaGraphicsResource_t cudaResource;
  cudaError_t cError;
  GLenum glError;

  //make a buffer for the handle if the user has forgotten too
  if(!glIsBuffer(openGLHandle))
    {
    glGenBuffers(1,&openGLHandle);
    glError = glGetError();
    if(glError != GL_NO_ERROR)
      {
      throw dax::cont::ErrorExecution(
            "could not generate an OpenGL buffer.");
      }
    }

  //bind the buffer to the given buffer type
  glBindBuffer(this->Type, openGLHandle);
  std::cerr << openGLHandle << std::endl;
  glError = glGetError();
  if(glError != GL_NO_ERROR)
    {

    std::cerr << "Could not bind to the given OpenGL buffer handle." << std::endl;
    throw dax::cont::ErrorExecution(
            "Could not bind to the given OpenGL buffer handle.");
    }

  //Allocate the memory and set it as GL_DYNAMIC_DRAW draw
  const std::size_t size = sizeof(ValueType)* handle.GetNumberOfValues();
  glBufferData(this->Type, size, 0, GL_DYNAMIC_DRAW);
  glError = glGetError();
  if(glError != GL_NO_ERROR)
    {
    throw dax::cont::ErrorControlOutOfMemory(
            "Could not allocate enough memory in OpenGL.");
    }

  //register the buffer as being used by cuda
  cError = cudaGraphicsGLRegisterBuffer(&cudaResource,
                                        openGLHandle,
                                        cudaGraphicsMapFlagsWriteDiscard);
  if(cError != cudaSuccess)
    {
    throw dax::cont::ErrorExecution(
            "Could not register the OpenGL buffer handle to CUDA.");
    }

  //map the resource into cuda, so we can copy it
  cError =cudaGraphicsMapResources(1,&cudaResource);
  if(cError != cudaSuccess)
    {
    throw dax::cont::ErrorControlOutOfMemory(
            "Could not allocate enough memory in CUDA for OpenGL interop.");
    }

  //get the mapped pointer
  std::size_t cuda_size;
  ValueType* beginPointer=NULL;
  cError = cudaGraphicsResourceGetMappedPointer((void **)&beginPointer,
                                       &cuda_size,
                                       cudaResource);

  if(cError != cudaSuccess)
    {
    throw dax::cont::ErrorExecution(
            "Unable to get pointers to CUDA memory for OpenGL buffer.");
    }

  //assert that cuda_size is the same size as the buffer we created in OpenGL
  DAX_ASSERT_CONT(cuda_size == size);

  //get the device pointers
  typedef dax::cont::ArrayHandle<ValueType,
    ContainerTag, DeviceAdapterTag> HandleType;
  typedef typename HandleType::PortalConstExecution PortalType;
  PortalType portal = handle.PrepareForInput();

  //Copy the data into memory that opengl owns, since we can't
  //give memory from cuda to opengl
  ::dax::thrust::cont::internal::CopyPortal(portal, beginPointer);

  //unmap the resource
  cudaGraphicsUnmapResources(1, &cudaResource);

  //unregister the buffer
  cudaGraphicsUnregisterResource(cudaResource);

  }
private:
  GLenum Type;
};



}
}
} //namespace dax::opengl::internal


#endif

