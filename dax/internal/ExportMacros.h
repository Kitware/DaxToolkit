/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax__internal__ExportMacros_h
#define __dax__internal__ExportMacros_h

/*!
  * Export macros for various parts of the Dax library.
  */

#ifdef __CUDACC__
#define DAX_CUDA
#endif

#ifdef DAX_CUDA
#define DAX_EXEC_EXPORT inline __device__
#define DAX_EXEC_CONT_EXPORT inline __device__ __host__
#else
#define DAX_EXEC_EXPORT inline
#define DAX_EXEC_CONT_EXPORT inline
#endif

// TODO: Do proper exports for dlls.
#define DAX_CONT_EXPORT

// Worklet macros.
#define DAX_WORKLET DAX_EXEC_EXPORT
#define DAX_IN
#define DAX_OUT

#endif //__dax__internal__ExportMacros_h
