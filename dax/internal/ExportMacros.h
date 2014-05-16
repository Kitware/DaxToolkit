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

#ifdef __CUDA_ARCH__
#define DAX_CUDA_COMPILATION
#endif

#ifdef _OPENMP
#define DAX_OPENMP
#endif

#ifdef DAX_CUDA
#define DAX_EXEC_EXPORT inline __device__
#define DAX_EXEC_CONT_EXPORT inline __device__ __host__
#define DAX_EXEC_CONSTANT_EXPORT __device__ __constant__
#else
#define DAX_EXEC_EXPORT inline
#define DAX_EXEC_CONT_EXPORT inline
#define DAX_EXEC_CONSTANT_EXPORT
#endif

#define DAX_CONT_EXPORT inline

/// Simple macro to identify a parameter as unused. This allows you to name a
/// parameter that is not used. There are several instances where you might
/// want to do this. For example, when using a parameter to overload or
/// template a function but do not actually use the parameter. Another example
/// is providing a specialization that does not need that parameter.
#define daxNotUsed(parameter_name)


// Check boost support under CUDA
#ifdef DAX_CUDA
#if !defined(BOOST_SP_DISABLE_THREADS) && !defined(BOOST_SP_USE_SPINLOCK) && !defined(BOOST_SP_USE_PTHREADS)
#warning -------------------------------------------------------------------
#warning The CUDA compiler (nvcc) has trouble with some of the optimizations
#warning boost uses for thread saftey.  To get around this, please define
#warning one of the following macros to specify the thread handling boost
#warning should use:
#warning
#warning   BOOST_SP_DISABLE_THREADS
#warning   BOOST_SP_USE_SPINLOCK
#warning   BOOST_SP_USE_PTHREADS
#warning
#warning Failure to define one of these for a CUDA build will probably cause
#warning other annoying warnings and might even cause incorrect code.  Note
#warning that specifying BOOST_SP_DISABLE_THREADS does not preclude using
#warning Dax with a threaded device (like OpenMP).  Specifying one of these
#warning modes for boost does not effect the scheduling in Dax.
#warning -------------------------------------------------------------------

#endif
#endif

#endif //__dax__internal__ExportMacros_h
