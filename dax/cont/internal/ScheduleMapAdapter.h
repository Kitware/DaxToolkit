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
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class FunctorType, class ExecutionArrayPortalType>
struct ScheduleMappingAdapter
{
  DAX_CONT_EXPORT ScheduleMappingAdapter(const FunctorType& functor,
                                         const ExecutionArrayPortalType &lookup)
    : Functor(functor), LookupTable(lookup) { }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const
  {
    //send the index as the key, and the LookupTable[index] as value

    this->Functor(index, this->LookupTable.Get(index));
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Functor.SetErrorMessageBuffer(errorMessage);
  }

private:
  FunctorType Functor;
  ExecutionArrayPortalType LookupTable;
};

template<class FunctorType, class ExecutionArrayPortalType>
struct ScheduleMappingAdapter2
{
  DAX_CONT_EXPORT ScheduleMappingAdapter2(const FunctorType& functor,
                                         const ExecutionArrayPortalType &lookup1,
                                         const ExecutionArrayPortalType &lookup2)
    : Functor(functor), LookupTable1(lookup1), LookupTable2(lookup2) { }

  DAX_EXEC_EXPORT void operator()(
      dax::Id index,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    //send the index as the key, and the LookupTable[index] as value

    this->Functor(this->LookupTable1.Get(index),
                  this->LookupTable2.Get(index),
                  errorMessage);
  }
private:
  FunctorType Functor;
  ExecutionArrayPortalType LookupTable1;
  ExecutionArrayPortalType LookupTable2;
};

}
}
}
} //namespace dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace internal {

template<class FunctorType,
         class Container,
         class Adapter>
DAX_CONT_EXPORT void ScheduleMap(
    FunctorType functor,
    const dax::cont::ArrayHandle<dax::Id,Container,Adapter> &values)
{
  //package up the ids to extract so we can do valid lookups
  const dax::Id size(values.GetNumberOfValues());

  typedef typename dax::cont::ArrayHandle<dax::Id,Container,Adapter>::
      PortalConstExecution PortalConstType;

  dax::exec::internal::kernel::ScheduleMappingAdapter<
      FunctorType,PortalConstType>
      mapFunctor(functor,
                 values.PrepareForInput());

  dax::cont::internal::Schedule(mapFunctor,
                                size,
                                Adapter());
}

template<class FunctorType,
         class Container,
         class Adapter>
DAX_CONT_EXPORT void ScheduleMap2(
    FunctorType functor,
    const dax::cont::ArrayHandle<dax::Id,Container,Adapter> &values1,
    const dax::cont::ArrayHandle<dax::Id,Container,Adapter> &values2)
{
  //package up the ids to extract so we can do valid lookups
  const dax::Id size(values1.GetNumberOfValues());

  typedef typename dax::cont::ArrayHandle<dax::Id,Container,Adapter>::
      PortalConstExecution PortalConstType;

  dax::exec::internal::kernel::ScheduleMappingAdapter2<
      FunctorType,PortalConstType>
      mapFunctor(functor,
                 values1.PrepareForInput(),
                 values2.PrepareForInput());

  dax::cont::internal::Schedule(mapFunctor,
                                size,
                                Adapter());
}
}}} // namespace dax::cont::internal

#endif // __dax_cont_internal_ScheduleSubgroupAdapter_h
