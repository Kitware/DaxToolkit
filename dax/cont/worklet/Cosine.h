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
#ifndef __dax_cont_worklet_Cosine_h
#define __dax_cont_worklet_Cosine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <Worklets/Cosine.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Cosine
{
  DAX_CONT_EXPORT
  Cosine(PortalType1 inValueArray, PortalType2 outValueArray)
    : InValueArray(inValueArray), OutValueArray(outValueArray) {  }

  template<class A, class B>
  DAX_EXEC_EXPORT void operator()(A, dax::Id index, B) const
  {
    const typename PortalType1::ValueType inValue =
        this->InValueArray.Get(index);
    typename PortalType2::ValueType outValue;
    dax::worklet::Cosine(inValue, outValue);
    this->OutValueArray.Set(index, outValue);
  }

private:
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

template<typename ValueType,
         class Container1,
         class Container2,
         class Adapter>
DAX_CONT_EXPORT void Cosine(
    const dax::cont::ArrayHandle<ValueType, Container1, Adapter> &inHandle,
    dax::cont::ArrayHandle<ValueType, Container2, Adapter> &outHandle)
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Cosine<
      typename dax::cont::ArrayHandle<ValueType,Container1,Adapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<ValueType,Container2,Adapter>::PortalExecution>
      kernel(inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel,
                                0,
                                fieldSize,
                                Container1(),
                                Adapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Cosine_h
