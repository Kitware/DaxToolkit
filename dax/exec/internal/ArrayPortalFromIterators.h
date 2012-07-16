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
#ifndef __dax_exec_internal_ArrayPortalFromIterators_h
#define __dax_exec_internal_ArrayPortalFromIterators_h

#include <dax/Types.h>

#include <iterator>

namespace dax {
namespace exec {
namespace internal {

/// This templated implementation of an ArrayPortal allows you to adapt a pair
/// of begin/end iterators to an ArrayPortal interface.
///
template<class IteratorT>
class ArrayPortalFromIterators
{
public:
  typedef IteratorT IteratorType;
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;

  DAX_EXEC_EXPORT ArrayPortalFromIterators() {  }

  DAX_EXEC_EXPORT
  ArrayPortalFromIterators(IteratorType begin, IteratorType end)
    : BeginIterator(begin), EndIterator(end) {  }

  DAX_EXEC_EXPORT
  dax::Id GetNumberOfValues() const {
    // Not using std::distance because on CUDA it cannot be used on a device.
    return (this->EndIterator - this->BeginIterator);
  }

  DAX_EXEC_EXPORT
  ValueType Get(dax::Id index) const {
    return *this->IteratorAt(index);
  }

  DAX_EXEC_EXPORT
  void Set(dax::Id index, ValueType value) const {
    *this->IteratorAt(index) = value;
  }

  DAX_EXEC_EXPORT
  IteratorType GetIteratorBegin() const { return this->BeginIterator; }

  DAX_EXEC_EXPORT
  IteratorType GetIteratorEnd() const { return this->EndIterator; }

private:
  IteratorType BeginIterator;
  IteratorType EndIterator;

  DAX_EXEC_EXPORT
  IteratorType IteratorAt(dax::Id index) const {
    // Not using std::advance because on CUDA it cannot be used on a device.
    return (this->BeginIterator + index);
  }
};

}
}
} // namespace dax::exec::internal

#endif //__dax_exec_internal_ArrayPortalFromIterators_h
