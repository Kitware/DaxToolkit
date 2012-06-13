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
#ifndef __dax__internal__ExportMacros_h
#define __dax__internal__ExportMacros_h

/*!
  * Export macros for various parts of the Dax library.
  */

#ifdef __CUDACC__
#define DAX_CUDA
#endif

#ifdef _OPENMP
#define DAX_OPENMP
#endif

#ifdef DAX_CUDA
#define DAX_EXEC_EXPORT inline __device__
#define DAX_EXEC_CONT_EXPORT inline __device__ __host__
#else
#define DAX_EXEC_EXPORT inline
#define DAX_EXEC_CONT_EXPORT inline
#endif

#define DAX_CONT_EXPORT inline

// Worklet macros.
#define DAX_WORKLET DAX_EXEC_EXPORT

#endif //__dax__internal__ExportMacros_h
