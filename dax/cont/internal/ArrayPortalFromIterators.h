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
#ifndef __dax_cont_internal_ArrayPortalFromIterators_h
#define __dax_cont_internal_ArrayPortalFromIterators_h

#include <dax/Types.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/Assert.h>

#include <iterator>

namespace dax {
namespace cont {
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

  DAX_CONT_EXPORT ArrayPortalFromIterators() {  }

  DAX_CONT_EXPORT
  ArrayPortalFromIterators(IteratorType begin, IteratorType end)
    : BeginIterator(begin), EndIterator(end)
  {
    DAX_ASSERT_CONT(this->GetNumberOfValues() >= 0);
  }

  /// Copy constructor for any other ArrayPortalFromIterators with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherIteratorT>
  DAX_CONT_EXPORT
  ArrayPortalFromIterators(const ArrayPortalFromIterators<OtherIteratorT> &src)
    : BeginIterator(src.GetIteratorBegin()),
      EndIterator(src.GetIteratorEnd())
  {  }

  template<typename OtherIteratorT>
  DAX_CONT_EXPORT
  ArrayPortalFromIterators<IteratorType> &operator=(
      const ArrayPortalFromIterators<OtherIteratorT> &src)
  {
    this->BeginIterator = src.GetIteratorBegin();
    this->EndIterator = src.GetIteratorBegin();
    return *this;
  }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    return std::distance(this->BeginIterator, this->EndIterator);
  }

  DAX_CONT_EXPORT
  ValueType Get(dax::Id index) const {
    return *this->IteratorAt(index);
  }

  DAX_CONT_EXPORT
  void Set(dax::Id index, ValueType value) const {
    *this->IteratorAt(index) = value;
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const { return this->BeginIterator; }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const { return this->EndIterator; }

private:
  IteratorType BeginIterator;
  IteratorType EndIterator;

  DAX_CONT_EXPORT
  IteratorType IteratorAt(dax::Id index) const {
    DAX_ASSERT_CONT(index >= 0);
    DAX_ASSERT_CONT(index < this->GetNumberOfValues());

    return this->BeginIterator + index;
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayPortalFromIterators_h
