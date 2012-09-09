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

#include <dax/cont/arg/FieldArrayHandle.h>
#include <dax/cont/Schedule.h>

namespace dax {
namespace cont {
namespace worklet {

template<class Container1,
         class Container2,
         class DeviceAdapter>
DAX_CONT_EXPORT void Elevation(
    const dax::cont::ArrayHandle<dax::Vector3,Container1,DeviceAdapter> &inHandle,
    dax::cont::ArrayHandle<dax::Scalar,Container2,DeviceAdapter> &outHandle,
    const dax::Vector3 &lowPoint = dax::make_Vector3(0.0, 0.0, 0.0),
    const dax::Vector3 &highPoint = dax::make_Vector3(0.0, 0.0, 1.0),
    const dax::Vector2 &outputRange = dax::make_Vector2(0.0, 1.0))
{
  dax::worklet::Elevation elev(lowPoint,highPoint,outputRange);
  dax::cont::Schedule<DeviceAdapter>(elev,inHandle,outHandle);
}


}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Elevation_h
