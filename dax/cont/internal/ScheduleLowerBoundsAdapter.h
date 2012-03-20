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


#include <iostream>
#include <boost/timer.hpp>

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

template<class Functor>
struct ScheduleLowerBoundsMapAdapter
{
  DAX_CONT_EXPORT ScheduleLowerBoundsMapAdapter(const Functor& functor,
                                          dax::internal::DataArray<dax::Id> values,
                                          dax::internal::DataArray<dax::Id> lookup)
    : Function(functor), LowerBoundValues(values), LookupTable(lookup) { }

  template<class Parameters, class ErrorHandler>
  DAX_EXEC_EXPORT void operator()(Parameters parameters, dax::Id index,
                                  ErrorHandler& errorHandler )
  {

    this->Function(parameters,
                   this->LowerBoundValues.GetValue(index)-1,
                   this->LookupTable.GetValue(index),
                   errorHandler);
  }
private:
  Functor Function;
  dax::internal::DataArray<dax::Id> LowerBoundValues;
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
DAX_CONT_EXPORT void ScheduleLowerBounds(
    Functor functor,
    Parameters parameters,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> values)
{
  typedef dax::exec::kernel::internal::ScheduleLowerBoundsMapAdapter<Functor> LowerBoundsFunctor;

  boost::timer bt;
  bt.restart();

  const dax::Id size(values.GetNumberOfEntries());
  //create a temporary list of all the values from 1 to size+1
  //this is the new cell index off by 1
  dax::cont::ArrayHandle<dax::Id,DeviceAdapter> lowerBoundsResult(size);
  DeviceAdapter::Schedule(dax::exec::kernel::internal::ScheduleInc(),
                          lowerBoundsResult.ReadyAsOutput(),
                          size);
  lowerBoundsResult.CompleteAsOutput();
  std::cout << "    ScheduleInc time " << bt.elapsed() << std::endl;
  bt.restart();

  //use the new cell index array and the values array to generate
  //the lower bounds array
  DeviceAdapter::LowerBounds(values,lowerBoundsResult,lowerBoundsResult);
  lowerBoundsResult.CompleteAsOutput();
  std::cout << "    Lower Bounds " << bt.elapsed() << std::endl;
  bt.restart();

  //with those two arrays we can now schedule the functor on each of those
  //items where me map
  LowerBoundsFunctor lowerBoundsFunctor(functor,
                                        values.ReadyAsInput(),
                                        lowerBoundsResult.ReadyAsInput());

  DeviceAdapter::Schedule(lowerBoundsFunctor,parameters,size);
  std::cout << "    Topology Schedule Call time  " << bt.elapsed() << std::endl;
  bt.restart();
}

}}}



#endif // __dax_cont_internal_ScheduleLowerBounds_h
