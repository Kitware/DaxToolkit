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

#ifndef __dax__opengl__internal_OpenGLHeaders_h
#define __dax__opengl__internal_OpenGLHeaders_h

#include <dax/internal/ExportMacros.h>

#if defined(__APPLE__)
# include <GL/glew.h>
# include <OpenGL/gl.h>
#else
# include <GL/glew.h>
# include <GL/gl.h>
#endif

#ifdef DAX_CUDA
# include <cuda_runtime.h>
# include <cuda_gl_interop.h>
#endif




#endif //__dax__opengl__internal_OpenGLHeaders_h
