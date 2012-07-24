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
#ifndef __dax_cont_worklet_Square_h
#define __dax_cont_worklet_Square_h

// TODO: This should be auto-generated.

#include <Worklets/Square.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Square
{
  DAX_CONT_EXPORT
  Square(const dax::worklet::Square &worklet,
         const PortalType1 &inValueArray,
         const PortalType2 &outValueArray)
    : Worklet(worklet),
      InValueArray(inValueArray),
      OutValueArray(outValueArray) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id index,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::Square &constWorklet = this->Worklet;

    const typename PortalType1::ValueType inValue =
        this->InValueArray.Get(index);
    typename PortalType2::ValueType outValue;

    constWorklet(inValue, outValue);

    this->OutValueArray.Set(index, outValue);
  }
private:
  dax::worklet::Square Worklet;
  const PortalType1 &InValueArray;
  const PortalType2 &OutValueArray;
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {

template<typename ValueType,
         class Container1,
         class Container2,
         class DeviceAdapter>
DAX_CONT_EXPORT void Square(
    const dax::cont::ArrayHandle<ValueType,Container1,DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<ValueType,Container2,DeviceAdapter> &outHandle)
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Square<
      typename dax::cont::ArrayHandle<ValueType,Container1,DeviceAdapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<ValueType,Container2,DeviceAdapter>::PortalExecution>
      kernel(dax::worklet::Square(),
             inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel,
                                fieldSize,
                                DeviceAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Square_h
