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

#ifndef __dax__opengl__internal_BufferTypePicker_h
#define __dax__opengl__internal_BufferTypePicker_h

#include <dax/Types.h>
#include <dax/opengl/internal/OpenGLHeaders.h>

namespace dax {
namespace opengl {
namespace internal {

/// helper function that guesses what OpenGL buffer type is the best default
/// given a primitive type. Currently GL_ELEMENT_ARRAY_BUFFER is used for integer
/// types, and GL_ARRAY_BUFFER is used for everything else
DAX_CONT_EXPORT GLenum BufferTypePicker( int )
{ return GL_ELEMENT_ARRAY_BUFFER; }

DAX_CONT_EXPORT GLenum BufferTypePicker( unsigned int )
{ return GL_ELEMENT_ARRAY_BUFFER; }

#if DAX_SIZE_LONG == 8

DAX_CONT_EXPORT GLenum BufferTypePicker( dax::internal::Int64Type )
{ return GL_ELEMENT_ARRAY_BUFFER; }

DAX_CONT_EXPORT GLenum BufferTypePicker( dax::internal::UInt64Type )
{ return GL_ELEMENT_ARRAY_BUFFER; }

#endif

template<typename T>
DAX_CONT_EXPORT GLenum BufferTypePicker( T )
{ return GL_ARRAY_BUFFER; }


}
}
} //namespace dax::opengl::internal

#endif //__dax__opengl__internal_BufferTypePicker_h
