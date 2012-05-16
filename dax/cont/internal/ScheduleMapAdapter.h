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

#ifndef __dax_cont_internal_ScheduleSubgroupAdapter_h
#define __dax_cont_internal_ScheduleSubgroupAdapter_h

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class Functor, class ExecutionIteratorType>
struct ScheduleMappingAdapter
{
  DAX_CONT_EXPORT ScheduleMappingAdapter(const Functor& functor,
                                         ExecutionIteratorType lookup)
    : Function(functor), LookupTable(lookup) { }

  template<class Parameters, class ErrorHandler>
  DAX_EXEC_EXPORT void operator()(Parameters parameters,
                                  dax::Id index,
                                  ErrorHandler& errorHandler )
  {
    //send the index as the key, and the LookupTable[index] as value

    this->Function(parameters,
                   index,
                   *(this->LookupTable + index),
                   errorHandler);
  }
private:
  const Functor Function;
  const ExecutionIteratorType LookupTable;
};

}
}
}
} //namespace dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace internal {

template<class Functor,
         class Parameters,
         template <typename> class ArrayContainerControl,
         class DeviceAdapter>
DAX_CONT_EXPORT void ScheduleMap(
    Functor functor,
    Parameters parameters,
    dax::cont::ArrayHandle<dax::Id,ArrayContainerControl,DeviceAdapter> values)
{
  //package up the ids to extract so we can do valid lookups
  const dax::Id size(values.GetNumberOfValues());

  typedef typename DeviceAdapter
      ::template ExecutionAdapter<ArrayContainerControl> ExecutionAdapter;
  typedef typename ExecutionAdapter
      ::template FieldStructures<dax::Id>::IteratorType IteratorType;

  dax::exec::internal::kernel::ScheduleMappingAdapter<Functor, IteratorType>
      mapFunctor(
        functor,
        values.PrepareForInput().first);

  DeviceAdapter::Schedule(mapFunctor,parameters,size);
}

}}} // namespace dax::cont::internal

#endif // __dax_cont_internal_ScheduleSubgroupAdapter_h
