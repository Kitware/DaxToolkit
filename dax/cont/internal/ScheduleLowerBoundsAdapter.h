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
                                          dax::internal::DataArray<dax::Id> nci)
    : Function(functor), NewCellIndices(nci){ }

  template<class Parameters, class ErrorHandler>
  DAX_EXEC_EXPORT void operator()(Parameters parameters, dax::Id index,
                                  ErrorHandler& errorHandler )
  {    
    this->Function(parameters,
                   this->NewCellIndices.GetValue(index),
                   index,
                   errorHandler);
  }
private:
  Functor Function;
  dax::internal::DataArray<dax::Id> NewCellIndices;
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
    const dax::Id size,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> cellCounts)
{
  typedef dax::exec::kernel::internal::ScheduleLowerBoundsMapAdapter<Functor> LowerBoundsFunctor;


  std::cout << "cell count is: " << size << std::endl;
  //create a temporary list of all the values from 1 to size+1
  //this is the new cell index off by 1
  dax::cont::ArrayHandle<dax::Id,DeviceAdapter> newCellIndices(size);
  DeviceAdapter::Schedule(dax::exec::kernel::internal::ScheduleInc(),
                          newCellIndices.ReadyAsOutput(),
                          size);
  newCellIndices.CompleteAsOutput();

  //use the new cell index array and the values array to generate
  //the lower bounds array
  DeviceAdapter::LowerBounds(cellCounts,newCellIndices,newCellIndices);
  newCellIndices.CompleteAsOutput();

  //the lower bounds now holds the old index that each new cell maps
  //too, where the array index is the new cell index. This is the inverse
  //of what SchedulMapAdapter expects.
  LowerBoundsFunctor lowerBoundsFunctor(functor,
                                        newCellIndices.ReadyAsInput());

  DeviceAdapter::Schedule(lowerBoundsFunctor,parameters,size);
}

}}}



#endif // __dax_cont_internal_ScheduleLowerBounds_h
