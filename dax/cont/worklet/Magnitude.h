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
#ifndef __dax_cont_worklet_Magnitude_h
#define __dax_cont_worklet_Magnitude_h

// TODO: This should be auto-generated.

#include <Worklets/Magnitude.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Magnitude
{
  DAX_CONT_EXPORT
  Magnitude(const dax::worklet::Magnitude &worklet,
            const PortalType1 &inValueArray,
            const PortalType2 &outValueArray)
    : Worklet(worklet),
      InValueArray(inValueArray),
      OutValueArray(outValueArray) {  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const
  {
    const typename PortalType1::ValueType inValue =
        this->InValueArray.Get(index);
    typename PortalType2::ValueType outValue;

    this->Worklet(inValue, outValue);

    this->OutValueArray.Set(index, outValue);
  }

  DAX_CONT_EXPORT void SetErrorMessageBuffer(
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
  }

private:
  dax::worklet::Magnitude Worklet;
  PortalType1 InValueArray;
  PortalType2 OutValueArray;
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class Container1,
         class Container2,
         class DeviceAdapter>
DAX_CONT_EXPORT void Magnitude(
    const dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<dax::Scalar,Container2,DeviceAdapter> &outHandle)
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Magnitude<
      typename dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<dax::Scalar,Container2,DeviceAdapter>::PortalExecution>
      kernel(dax::worklet::Magnitude(),
             inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel,
                                fieldSize,
                                DeviceAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Magnitude_h
