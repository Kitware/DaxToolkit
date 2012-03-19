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

#ifndef __dax_cont_internal_ScheduleLowerBounds_h
#define __dax_cont_internal_ScheduleLowerBounds_h

#include <dax/Types.h>
#include <dax/internal/DataArray.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ScheduleMapAdapter.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {
struct ScheduleInc
{
DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                dax::Id index,
                                dax::exec::internal::ErrorHandler &)
{
  array.SetValue(index, index+1);
}
};

}
}
}
} //namespace dax::exec::kernel::internal

namespace dax {
namespace cont {
namespace internal {


template<class Functor, class Parameters, class DeviceAdapter>
DAX_CONT_EXPORT void ScheduleLowerBounds(
    Functor functor,
    Parameters parameters,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> values)
{
  const dax::Id size(values.GetNumberOfEntries());

  //create a temporary list of all the values from 1 to size+1
  //place that in lower bounds
  dax::cont::ArrayHandle<dax::Id,DeviceAdapter> lowerBoundsResult(size);
  DeviceAdapter::Schedule(dax::exec::kernel::internal::ScheduleInc(),
                          lowerBoundsResult.ReadyAsOutput(),
                          size);
  dax::cont::internal::ScheduleMap(functor,parameters,lowerBoundsResult);
}

}}}



#endif // __dax_cont_internal_ScheduleLowerBounds_h
