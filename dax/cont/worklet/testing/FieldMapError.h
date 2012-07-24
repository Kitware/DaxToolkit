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
#ifndef __dax_cont_worklet_testing_FieldMapError_h
#define __dax_cont_worklet_testing_FieldMapError_h

// TODO: This should be auto-generated.

#include <Worklets/Testing/FieldMapError.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/exec/internal/FieldAccess.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType>
struct FieldMapError
{
  DAX_CONT_EXPORT
  FieldMapError(const dax::worklet::testing::FieldMapError &worklet,
                const PortalType &inArray)
    : Worklet(worklet), InArray(inArray) {  }

  DAX_EXEC_EXPORT void operator()(
      dax::Id pointIndex,
      const dax::exec::internal::ErrorMessageBuffer &errorMessage)
  {
    this->Worklet.SetErrorMessageBuffer(errorMessage);
    const dax::worklet::testing::FieldMapError &constWorklet = this->Worklet;

    constWorklet(
        dax::exec::internal::FieldGet(this->InArray, pointIndex, constWorklet));
  }

private:
  dax::worklet::testing::FieldMapError Worklet;
  const PortalType &InArray;
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {
namespace testing {

template<typename ValueType,
         class Container,
         class Adapter>
inline void FieldMapError(
    const dax::cont::ArrayHandle<ValueType,Container,Adapter> &inArray)
{
  dax::Id fieldSize = inArray.GetNumberOfValues();

  dax::exec::internal::kernel::FieldMapError<
      typename dax::cont::ArrayHandle<ValueType,Container,Adapter>::PortalConstExecution>
      kernel(dax::worklet::testing::FieldMapError(),
             inArray.PrepareForInput());

  dax::cont::internal::Schedule(kernel, fieldSize, Adapter());
}

}
}
}
} //dax::cont::worklet::testing

#endif //__dax_cont_worklet_testing_FieldMapError_h
