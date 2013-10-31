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
#ifndef __dax_cont_internal_IteratorFromArrayPortal_h
#define __dax_cont_internal_IteratorFromArrayPortal_h

#include <dax/cont/ArrayPortal.h>

#include <dax/cont/Assert.h>

#include <boost/iterator/iterator_facade.hpp>

namespace dax {
namespace cont {
namespace internal {
namespace detail {

template<class ArrayPortalType>
struct IteratorFromArrayPortalValue {
  typedef typename ArrayPortalType::ValueType ValueType;

  DAX_CONT_EXPORT
  IteratorFromArrayPortalValue(const ArrayPortalType &portal, dax::Id index)
    : Portal(portal), Index(index) {  }

  DAX_CONT_EXPORT
  void Swap( IteratorFromArrayPortalValue<ArrayPortalType> &rhs ) throw()
  {
    //we need use the explicit type not a proxy temp object
    //A proxy temp object would point to the same underlying data structure
    //and would not hold the old value of *this once *this was set to rhs.
    const ValueType aValue = *this;
    *this = rhs;
    rhs = aValue;
  }

  DAX_CONT_EXPORT
  IteratorFromArrayPortalValue<ArrayPortalType> &operator=(
      const IteratorFromArrayPortalValue<ArrayPortalType> &rhs)
  {
    this->Portal.Set(this->Index, rhs.Portal.Get(rhs.Index));
    return *this;
  }

  DAX_CONT_EXPORT
  ValueType operator=(ValueType value) {
    this->Portal.Set(this->Index, value);
    return value;
  }

  DAX_CONT_EXPORT
  operator ValueType(void) const {
    return this->Portal.Get(this->Index);
  }

  const ArrayPortalType& Portal;
  dax::Id Index;
};

} // namespace detail

template<class ArrayPortalType>
class IteratorFromArrayPortal : public
    boost::iterator_facade<
      IteratorFromArrayPortal<ArrayPortalType>,
      typename ArrayPortalType::ValueType,
      boost::random_access_traversal_tag,
      detail::IteratorFromArrayPortalValue<ArrayPortalType>,
      dax::Id>
{
public:
  IteratorFromArrayPortal()
    : Portal(), Index(0) { }

  explicit IteratorFromArrayPortal(const ArrayPortalType &portal,
                                   dax::Id index = 0)
    : Portal(portal), Index(index) {  }

  DAX_CONT_EXPORT
  detail::IteratorFromArrayPortalValue<ArrayPortalType>
  operator[](int idx) const
  {
  return detail::IteratorFromArrayPortalValue<ArrayPortalType>(this->Portal,
                                                               idx);
  }

private:
  ArrayPortalType Portal;
  dax::Id Index;

  // Implementation for boost iterator_facade
  friend class boost::iterator_core_access;

  DAX_CONT_EXPORT
  detail::IteratorFromArrayPortalValue<ArrayPortalType> dereference() const {
    return detail::IteratorFromArrayPortalValue<ArrayPortalType>(this->Portal,
                                                                 this->Index);
  }

  DAX_CONT_EXPORT
  bool equal(const IteratorFromArrayPortal<ArrayPortalType> &other) const {
    // Technically, we should probably check that the portals are the same,
    // but the portal interface does not specify an equal operator.  It is
    // by its nature undefined what happens when comparing iterators from
    // different portals anyway.
    return (this->Index == other.Index);
  }

  DAX_CONT_EXPORT
  void increment() {
    this->Index++;
    DAX_ASSERT_CONT(this->Index >= 0);
    DAX_ASSERT_CONT(this->Index <= this->Portal.GetNumberOfValues());
  }

  DAX_CONT_EXPORT
  void decrement() {
    this->Index--;
    DAX_ASSERT_CONT(this->Index >= 0);
    DAX_ASSERT_CONT(this->Index <= this->Portal.GetNumberOfValues());
  }

  DAX_CONT_EXPORT
  void advance(dax::Id delta) {
    this->Index += delta;
    DAX_ASSERT_CONT(this->Index >= 0);
    DAX_ASSERT_CONT(this->Index <= this->Portal.GetNumberOfValues());
  }

  DAX_CONT_EXPORT
  dax::Id
  distance_to(const IteratorFromArrayPortal<ArrayPortalType> &other) const {
    // Technically, we should probably check that the portals are the same,
    // but the portal interface does not specify an equal operator.  It is
    // by its nature undefined what happens when comparing iterators from
    // different portals anyway.
    return other.Index - this->Index;
  }
};

template<class ArrayPortalType>
IteratorFromArrayPortal<ArrayPortalType> make_IteratorBegin(
    const ArrayPortalType &portal)
{
  return IteratorFromArrayPortal<ArrayPortalType>(portal, 0);
}

template<class ArrayPortalType>
IteratorFromArrayPortal<ArrayPortalType> make_IteratorEnd(
    const ArrayPortalType &portal)
{
  return IteratorFromArrayPortal<ArrayPortalType>(portal,
                                                  portal.GetNumberOfValues());
}


//implementat a custom swap function, since the std::swap won't work
//since we return RValues instead of Lvalues
template<typename T>
  void swap( dax::cont::internal::detail::IteratorFromArrayPortalValue<T> a,
             dax::cont::internal::detail::IteratorFromArrayPortalValue<T> b)
  {
    a.Swap(b);
  }


}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_IteratorFromArrayPortal_h
