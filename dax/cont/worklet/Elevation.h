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

#include <Worklets/Elevation.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class PortalType1, class PortalType2>
struct Elevation
{
  DAX_CONT_EXPORT
  Elevation(const dax::worklet::Elevation &worklet,
            PortalType1 inCoordinates,
            PortalType2 outField)
    : Worklet(worklet), InCoordinates(inCoordinates), OutField(outField) {  }

  DAX_EXEC_EXPORT
  void operator()(dax::Id index,
                  const dax::exec::internal::ErrorMessageBuffer &errorBuffer)
  {
    this->Worklet.SetErrorMessageBuffer(errorBuffer);
    const dax::worklet::Elevation &constWorklet = this->Worklet;

    const typename PortalType1::ValueType inCoordinates =
        this->InCoordinates.Get(index);
    typename PortalType2::ValueType outField;

    constWorklet(inCoordinates, outField);

    this->OutField.Set(index, outField);
  }
private:
  dax::worklet::Elevation Worklet;
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
    dax::cont::ArrayHandle<dax::Scalar, Container2, Adapter> &outHandle,
    const dax::Vector3 &lowPoint = dax::make_Vector3(0.0, 0.0, 0.0),
    const dax::Vector3 &highPoint = dax::make_Vector3(0.0, 0.0, 1.0),
    const dax::Vector2 &outputRange = dax::make_Vector2(0.0, 1.0))
{
  dax::Id fieldSize = inHandle.GetNumberOfValues();

  dax::exec::internal::kernel::Elevation<
      typename dax::cont::ArrayHandle<dax::Vector3,Container1,Adapter>::PortalConstExecution,
      typename dax::cont::ArrayHandle<dax::Scalar,Container2,Adapter>::PortalExecution>
      kernel(dax::worklet::Elevation(lowPoint, highPoint, outputRange),
             inHandle.PrepareForInput(),
             outHandle.PrepareForOutput(fieldSize));

  dax::cont::internal::Schedule(kernel, fieldSize, Adapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
