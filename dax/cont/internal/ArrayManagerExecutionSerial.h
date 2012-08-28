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
#ifndef __dax_cont_internal_ArrayManagerExecutionSerial_h
#define __dax_cont_internal_ArrayManagerExecutionSerial_h

#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/cont/internal/ArrayManagerExecutionShareWithControl.h>
#include <dax/cont/internal/DeviceAdapterTagSerial.h>

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerControlTag>
class ArrayManagerExecution
    <T, ArrayContainerControlTag, dax::cont::DeviceAdapterTagSerial>
    : public dax::cont::internal::ArrayManagerExecutionShareWithControl
          <T, ArrayContainerControlTag>
{
public:
  typedef dax::cont::internal::ArrayManagerExecutionShareWithControl
      <T, ArrayContainerControlTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayManagerExecutionSerial_h
