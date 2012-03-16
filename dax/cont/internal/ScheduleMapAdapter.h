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
#include <dax/internal/DataArray.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {


template<class Functor>
struct ScheduleMappingAdapter
{
  DAX_CONT_EXPORT ScheduleMappingAdapter(const Functor& functor,
                                          dax::internal::DataArray<dax::Id> lookup)
    : Function(functor), LookupTable(lookup) { }

  template<class Parameters, class ErrorHandler>
  DAX_EXEC_EXPORT void operator()(Parameters parameters, dax::Id index,
                                  ErrorHandler& errorHandler )
  {
    //send the index as the key, and the LookupTable[index] as value

    this->Function(parameters,
                   index,
                   this->LookupTable.GetValue(index),
                   errorHandler);
  }
private:
  Functor Function;
  dax::internal::DataArray<dax::Id> LookupTable;
};

}
}
}
} //namespace dax::exec::kernel::internal

namespace dax {
namespace cont {
namespace internal {


template<class Functor, class Parameters, class DeviceAdapter>
DAX_CONT_EXPORT void ScheduleMap(
    Functor functor,
    Parameters parameters,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> values)
{
  //package up the ids to extract so we can do valid lookups
  const dax::Id size(values.GetNumberOfEntries());

  dax::exec::kernel::internal::ScheduleMappingAdapter<Functor> mapFunctor(
                                      functor,values.ReadyAsInput());

  DeviceAdapter::Schedule(mapFunctor,parameters,size);
}

}}}

#endif // __dax_cont_internal_ScheduleSubgroupAdapter_h
