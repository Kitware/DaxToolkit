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
#ifndef __dax_cont_DeviceAdapter_h
#define __dax_cont_DeviceAdapter_h

// These are listed in non-alphabetical order because this is the conceptual
// order in which the sub-files are loaded.  (But the compile should still
// succeed of the order is changed.)

#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY
/// \brief A tag specifying the interface between the control and execution environments.
///
/// A DeviceAdapter tag specifies a set of functions and classes that provide
/// mechanisms to run algorithms on a type of parallel device. The tag
/// DeviceAdapterTag___ does not actually exist. Rather, this documentation is
/// provided to describe the interface for a DeviceAdapter. Loading the
/// dax/cont/DeviceAdapter.h header file will set a default device adapter
/// appropriate for the current compile environment. You can specify the
/// default device adapter by first setting the \c DAX_DEVICE_ADAPTER macro.
/// Valid values for \c DAX_DEVICE_ADAPTER are the following:
///
/// \li \c DAX_DEVICE_ADAPTER_SERIAL Runs all algorithms in serial. Can be
/// helpful for debugging.
/// \li \c DAX_DEVICE_ADAPTER_CUDA Schedules and runs algorithms on a GPU
/// using CUDA.  Must be compiling with a CUDA compiler (nvcc).
/// \li \c DAX_DEVICE_ADAPTER_OPENMP Schedules an algorithm over multiple
/// CPU cores using OpenMP compiler directives.  Must be compiling with an
/// OpenMP-compliant compiler with OpenMP pragmas enabled.
/// \li \c DAX_DEVICE_ADAPTER_TBB Schedule and runs algorithms on multiple
/// threads using the Intel Threading Building Blocks (TBB) libraries. Must
/// have the TBB headers available and the resulting code must be linked with
/// the TBB libraries.
///
/// See the ArrayManagerExecution.h and DeviceAdapterAlgorithm.h files for
/// documentation on all the functions and classes that must be
/// overloaded/specialized to create a new device adapter.
///
struct DeviceAdapterTag___ {  };
#endif //DAX_DOXYGEN_ONLY

namespace internal {

} // namespace internal

}
} // namespace dax::cont


#endif //__dax_cont_DeviceAdapter_h
