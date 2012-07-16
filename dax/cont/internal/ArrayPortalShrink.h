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
#ifndef __dax_cont_internal_ArrayPortalShrink_h
#define __dax_cont_internal_ArrayPortalShrink_h

#include <dax/Types.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/Assert.h>

namespace dax {
namespace cont {
namespace internal {

/// This ArrayPortal adapter is a utility that allows you to shrink the
/// (reported) array size without actually modifying the underlying allocation.
///
template<class PortalT>
class ArrayPortalShrink
{
public:
  typedef PortalT DelegatePortalType;

  typedef typename DelegatePortalType::ValueType ValueType;
  typedef typename DelegatePortalType::IteratorType IteratorType;

  DAX_CONT_EXPORT ArrayPortalShrink() : NumberOfValues(0) {  }

  DAX_CONT_EXPORT ArrayPortalShrink(const DelegatePortalType &delegatePortal)
    : DelegatePortal(delegatePortal),
      NumberOfValues(delegatePortal.GetNumberOfValues())
  {  }

  DAX_CONT_EXPORT ArrayPortalShrink(const DelegatePortalType &delegatePortal,
                                    dax::Id numberOfValues)
    : DelegatePortal(delegatePortal), NumberOfValues(numberOfValues)
  {
    DAX_ASSERT_CONT(numberOfValues <= delegatePortal.GetNumberOfValues());
  }

  /// Copy constructor for any other ArrayPortalShrink with a delegate type
  /// that can be copied to this type. This allows us to do any type casting
  /// the delegates can do (like the non-const to const cast).
  ///
  template<class OtherDelegateType>
  DAX_CONT_EXPORT
  ArrayPortalShrink(const ArrayPortalShrink<OtherDelegateType> &src)
    : DelegatePortal(src.GetDelegatePortal()),
      NumberOfValues(src.GetNumberOfValues())
  {  }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->NumberOfValues; }

  DAX_CONT_EXPORT
  ValueType Get(dax::Id index) const {
    DAX_ASSERT_CONT(index >= 0);
    DAX_ASSERT_CONT(index < this->GetNumberOfValues());
    return this->DelegatePortal.Get(index);
  }

  DAX_CONT_EXPORT
  void Set(dax::Id index, ValueType value) const {
    DAX_ASSERT_CONT(index >= 0);
    DAX_ASSERT_CONT(index < this->GetNumberOfValues());
    this->DelegatePortal.Set(index, value);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const {
    return this->DelegatePortal.GetIteratorBegin();
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const {
    IteratorType iterator = this->DelegatePortal.GetIteratorBegin();
    std::advance(iterator, this->GetNumberOfValues());
    return iterator;
  }

  /// Special method in this ArrayPortal that allows you to shrink the
  /// (exposed) array.
  ///
  DAX_CONT_EXPORT
  void Shrink(dax::Id numberOfValues) {
    DAX_ASSERT_CONT(numberOfValues < this->GetNumberOfValues());
    this->NumberOfValues = numberOfValues;
  }

  /// Get a copy of the delegate portal. Although safe, this is probably only
  /// useful internally. (It is exposed as public for the templated copy
  /// constructor.)
  ///
  DelegatePortalType GetDelegatePortal() const { return this->DelegatePortal; }

private:
  DelegatePortalType DelegatePortal;
  dax::Id NumberOfValues;
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayPortalShrink_h
