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
#ifndef __dax_openmp_cont_internal_ArrayManagerExecutionOpenMP_h
#define __dax_openmp_cont_internal_ArrayManagerExecutionOpenMP_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/openmp/cont/internal/DeviceAdapterTagOpenMP.h>

#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/thrust/cont/internal/ArrayManagerExecutionThrustShare.h>

// These must be placed in the dax::cont::internal namespace so that
// the template can be found.

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerTag>
class ArrayManagerExecution
    <T, ArrayContainerTag, dax::openmp::cont::DeviceAdapterTagOpenMP>
    : public dax::thrust::cont::internal::ArrayManagerExecutionThrustShare
        <T, ArrayContainerTag>
{
public:
  typedef dax::thrust::cont::internal::ArrayManagerExecutionThrustShare
      <T, ArrayContainerTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;

  typedef typename Superclass::ThrustIteratorType ThrustIteratorType;
  typedef typename Superclass::ThrustIteratorConstType ThrustIteratorConstType;
};

}
}
} // namespace dax::cont::internal


#endif //__dax_openmp_cont_internal_ArrayManagerExecutionOpenMP_h
