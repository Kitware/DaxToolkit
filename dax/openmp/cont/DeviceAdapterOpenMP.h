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

#ifndef __dax_openmp_cont_DeviceAdapterOpenMP_h
#define __dax_openmp_cont_DeviceAdapterOpenMP_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::openmp::cont::DeviceAdapterTagOpenMP

// Forward declaration (before suber DeviceAdapterTagThrust declared).
namespace dax {
namespace openmp {
namespace cont {
struct DeviceAdapterTagOpenMP;
}
}
}

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/thrust/cont/internal/ArrayManagerExecutionThrustShare.h>
#include <dax/thrust/cont/internal/DeviceAdapterThrust.h>

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
  typedef dax::thrust::cont::internal::ArrayManagerExecutionThrustShare
      <T, ArrayContainerTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::IteratorType IteratorType;
  typedef typename Superclass::IteratorConstType IteratorConstType;

  typedef typename Superclass::ThrustIteratorType ThrustIteratorType;
  typedef typename Superclass::ThrustIteratorConstType ThrustIteratorConstType;
};

template<class Functor, class Parameters, class Container>
DAX_CONT_EXPORT void Schedule(Functor functor,
                              Parameters parameters,
                              dax::Id numInstances,
                              Container,
                              dax::openmp::cont::DeviceAdapterTagOpenMP)
{
  dax::thrust::cont::internal::ScheduleThrust(
        functor,
        parameters,
        numInstances,
        Container(),
        dax::openmp::cont::DeviceAdapterTagOpenMP());
}

}
}
} // namespace dax::cont::internal

namespace dax {
namespace exec {
namespace internal {

template <class ArrayContainerControlTag>
class ExecutionAdapter<ArrayContainerControlTag,
                       dax::openmp::cont::DeviceAdapterTagOpenMP>
    : public dax::thrust::cont::internal::ExecutionAdapterThrust
        <ArrayContainerControlTag,dax::openmp::cont::DeviceAdapterTagOpenMP>
{
public:
  typedef dax::thrust::cont::internal::ExecutionAdapterThrust
      <ArrayContainerControlTag,dax::openmp::cont::DeviceAdapterTagOpenMP>
      Superclass;
  using Superclass::FieldStructures;

  DAX_EXEC_EXPORT ExecutionAdapter(char *messageBegin, char *messageEnd)
    : Superclass(messageBegin, messageEnd) {  }
};

}
}
} // namespace dax::exec::internal

#endif //__dax_openmp_cont_DeviceAdapterOpenMP_h
