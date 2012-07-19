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
#ifndef __dax_cont_worklet_Elevation_h
#define __dax_cont_worklet_Elevation_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <Worklets/Elevation.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Elevation
{
  DAX_CONT_EXPORT
  Elevation(PortalType1 inCoordinates, PortalType2 outField)
    : InCoordinates(inCoordinates), OutField(outField) {  }

  template<class A, class B>
  DAX_EXEC_EXPORT
  void operator()(A, dax::Id index, B) const
  {
    const typename PortalType1::ValueType inCoordinates =
        this->InCoordinates.Get(index);
    typename PortalType2::ValueType outField;
    dax::worklet::Elevation(inCoordinates, outField);
    this->OutField.Set(index, outField);
  }
private:
  PortalType1 InCoordinates;
  PortalType2 OutField;
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
         class Adapter>
DAX_CONT_EXPORT void Elevation(
    const dax::cont::ArrayHandle<dax::Vector3, Container1, Adapter> &inHandle,
    dax::cont::ArrayHandle<dax::Scalar, Container2, Adapter> &outHandle)
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Elevation<
      typename dax::cont::ArrayHandle<dax::Vector3,Container1,Adapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<dax::Scalar,Container2,Adapter>::PortalExecution>
      kernel(inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel, 0, fieldSize, Container1(), Adapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
