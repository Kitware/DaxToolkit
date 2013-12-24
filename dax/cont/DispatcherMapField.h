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
#ifndef __dax_cont_DispatcherMapField_h
#define __dax_cont_DispatcherMapField_h

#include <dax/Types.h>
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/cont/dispatcher/DispatcherBase.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/internal/ParameterPack.h>

namespace dax { namespace cont {

template <
  class WorkletType_,
  class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class DispatcherMapField :
  public dax::cont::dispatcher::DispatcherBase<
          DispatcherMapField< WorkletType_, DeviceAdapterTag_ >,
          dax::exec::WorkletMapField,
          WorkletType_,
          DeviceAdapterTag_ >
{

  typedef dax::cont::dispatcher::DispatcherBase< DispatcherMapField< WorkletType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletMapField,
                                                 WorkletType_,
                                                 DeviceAdapterTag_> Superclass;
  friend class dax::cont::dispatcher::DispatcherBase< DispatcherMapField< WorkletType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletMapField,
                                                 WorkletType_,
                                                 DeviceAdapterTag_>;

public:
  typedef WorkletType_ WorkletType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

  DAX_CONT_EXPORT DispatcherMapField() : Superclass(WorkletType())
    { }
  DAX_CONT_EXPORT DispatcherMapField(WorkletType worklet) : Superclass(worklet)
    { }

private:
  template<typename ParameterPackType>
  DAX_CONT_EXPORT void DoInvoke(WorkletType worklet,
                                ParameterPackType arguments) const
  {
    this->BasicInvoke(worklet, arguments);
  }

};

} } // namespace dax::cont

#endif //__dax_cont_DispatcherMapField_h
