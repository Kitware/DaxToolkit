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
#ifndef __dax_tbb_cont_internal_ArrayManagerExecutionTBB_h
#define __dax_tbb_cont_internal_ArrayManagerExecutionTBB_h

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>

#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/thrust/cont/internal/ArrayManagerExecutionThrustShare.h>

// These must be placed in the dax::cont::internal namespace so that
// the template can be found.

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerTag>
class ArrayManagerExecution
    <T, ArrayContainerTag, dax::tbb::cont::DeviceAdapterTagTBB>
    : public dax::thrust::cont::internal::ArrayManagerExecutionThrustShare
        <T, ArrayContainerTag>
{
public:
  typedef dax::thrust::cont::internal::ArrayManagerExecutionThrustShare
      <T, ArrayContainerTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;
};

}
}
} // namespace dax::cont::internal


#endif //__dax_tbb_cont_internal_ArrayManagerExecutionTBB_h
