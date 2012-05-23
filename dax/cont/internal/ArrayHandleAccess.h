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
#ifndef __dax_cont_internal_ArrayHandleAccess_h
#define __dax_cont_internal_ArrayHandleAccess_h

#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {
namespace internal {

/// Some device adapters need to access internal structures of their
/// ArrayManagerExecution to efficiently perform their algorithms. This class
/// gives access to that class embedded in an ArrayHandle.
///
struct ArrayHandleAccess
{
  template<class T, class Container, class Device>
  DAX_CONT_EXPORT static
  dax::cont::internal::ArrayManagerExecution<T,Container,Device>
  &GetArrayManagerExecution(dax::cont::ArrayHandle<T,Container,Device> &array)
  {
    DAX_ASSERT_CONT(array.Internals->ExecutionArrayValid);
    return array.Internals->ExecutionArray;
  }

  template<class T, class Container, class Device>
  DAX_CONT_EXPORT static
  const dax::cont::internal::ArrayManagerExecution<T,Container,Device>
  &GetArrayManagerExecution(
      const dax::cont::ArrayHandle<T,Container,Device> &array)
  {
    DAX_ASSERT_CONT(array.Internals->ExecutionArrayValid);
    return array.Internals->ExecutionArray;
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayHandleAccess_h
