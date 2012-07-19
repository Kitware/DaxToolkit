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

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Square.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Square
{
  Square(PortalType1 inValueArray, PortalType2 outValueArray)
    : InValueArray(inValueArray), OutValueArray(outValueArray) {  }

  template<class A, class B>
  DAX_EXEC_EXPORT void operator()(A, dax::Id index, B) const
  {
    const typename PortalType1::ValueType inValue =
        this->InValueArray.Get(index);
    typename PortalType2::ValueType outValue;
    dax::worklet::Square(inValue, outValue);
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

template<typename FieldType,
         class Container1,
         class Container2,
         class DeviceAdapter>
DAX_CONT_EXPORT void Square(
    const dax::cont::ArrayHandle<FieldType,Container1,DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<FieldType,Container2,DeviceAdapter> &outHandle)
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Square<
      typename dax::cont::ArrayHandle<FieldType,Container1,DeviceAdapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<FieldType,Container2,DeviceAdapter>::PortalExecution>
      kernel(inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel,
                                0,
                                fieldSize,
                                Container1(),
                                DeviceAdapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Square_h
