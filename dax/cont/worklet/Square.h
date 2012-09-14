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

#include <dax/worklets/Square.worklet>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Schedule.h>

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
  dax::cont::Schedule<DeviceAdapter>(dax::worklet::Square(),inHandle,outHandle);
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Square_h
