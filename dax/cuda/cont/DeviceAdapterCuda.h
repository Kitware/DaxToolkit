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

#ifndef __dax_cuda_cont_DeviceAdapterCuda_h
#define __dax_cuda_cont_DeviceAdapterCuda_h

// Declare DAX_DEFAULT_DEVICE_ADAPTER and the tag it points to before including
// other headers that may require it.

#include <dax/thrust/cont/internal/DeviceAdapterThrustTag.h>

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cuda::cont::DeviceAdapterTagCuda

namespace dax {
namespace cuda {
namespace cont {

/// A DeviceAdapter that uses Cuda.  To use this adapter, the code must be
/// compiled with nvcc.
///
struct DeviceAdapterTagCuda
    : public dax::thrust::cont::internal::DeviceAdapterTagThrust
{  };

}
}
} // namespace dax::cuda::cont

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/thrust/cont/internal/ArrayManagerExecutionThrustDevice.h>
#include <dax/thrust/cont/internal/DeviceAdapterThrust.h>

// These must be placed in the dax::cont::internal namespace so that
// the template can be found.

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerTag>
class ArrayManagerExecution
    <T, ArrayContainerTag, dax::cuda::cont::DeviceAdapterTagCuda>
    : public dax::thrust::cont::internal::ArrayManagerExecutionThrustDevice
        <T, ArrayContainerTag>
{
public:
  typedef dax::thrust::cont::internal::ArrayManagerExecutionThrustDevice
      <T, ArrayContainerTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;

  typedef typename Superclass::ThrustIteratorType ThrustIteratorType;
  typedef typename Superclass::ThrustIteratorConstType ThrustIteratorConstType;
};

template<class Functor, class Parameters, class Container>
DAX_CONT_EXPORT void Schedule(Functor functor,
                              Parameters parameters,
                              dax::Id numInstances,
                              Container,
                              dax::cuda::cont::DeviceAdapterTagCuda)
{
  dax::thrust::cont::internal::ScheduleThrust(
        functor,
        parameters,
        numInstances,
        Container(),
        dax::cuda::cont::DeviceAdapterTagCuda());
}

}
}
} // namespace dax::cont::internal

namespace dax {
namespace exec {
namespace internal {

template <class ArrayContainerControlTag>
class ExecutionAdapter<ArrayContainerControlTag,
                       dax::cuda::cont::DeviceAdapterTagCuda>
    : public dax::thrust::cont::internal::ExecutionAdapterThrust
        <ArrayContainerControlTag,dax::cuda::cont::DeviceAdapterTagCuda>
{
public:
  typedef dax::thrust::cont::internal::ExecutionAdapterThrust
      <ArrayContainerControlTag,dax::cuda::cont::DeviceAdapterTagCuda>
      Superclass;
  using Superclass::FieldStructures;

  DAX_EXEC_EXPORT ExecutionAdapter(char *messageBegin, char *messageEnd)
    : Superclass(messageBegin, messageEnd) {  }
};

}
}
} // namespace dax::exec::internal

#endif //__dax_cuda_cont_DeviceAdapterCuda_h
