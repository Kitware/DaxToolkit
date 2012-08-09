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
#ifndef __dax_openmp_cont_internal_DeviceAdapterTagOpenMP_h
#define __dax_openmp_cont_internal_DeviceAdapterTagOpenMP_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/thrust/cont/internal/DeviceAdapterTagThrust.h>

namespace dax {
namespace openmp {
namespace cont {

/// A DeviceAdapter that uses OpenMP.  To use this adapter, an OpenMP-compliant
/// compiler with OpenMP support turned on must be used (duh).
///
struct DeviceAdapterTagOpenMP
    : public dax::thrust::cont::internal::DeviceAdapterTagThrust
{  };

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_internal_DeviceAdapterTagOpenMP_h
