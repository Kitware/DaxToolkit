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
#ifndef __dax_cont_ArrayContainerControlPermutation_h
#define __dax_cont_ArrayContainerControlPermutation_h

#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {

/// \brief An permutation array portal wraps and permutes two key-value portals.
///
/// The array portal comprises of two arrays portals one with the Key and other
/// with the Value. The Key holds the index into Value array and returns the
/// corresponding value. So for example if we have a permutation array portal
/// called PermuteArray with portal Key = [0,2,3,5] and portal Value =
/// [8,6,4,9,8,3]. Then
///
/// PermuteArray[0] = Value[Key[0]] = Value[0] = 8.
/// PermuteArray[1] = Value[Key[1]] = Value[2] = 4.
/// PermuteArray[2] = Value[Key[2]] = Value[3] = 9. etc ...
///
/// The ArrayPortalPermutation is used in an ArrayHandle with an
/// ArrayContainerControlTagPermutation container.
///
template <class KClassType,class VClassType>
class ArrayPortalPermutation
{
public:
  typedef KClassType KeyPortalType;
  typedef VClassType ValuePortalType;
  typedef typename ValuePortalType::ValueType ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalPermutation() {}

  DAX_EXEC_CONT_EXPORT
  ArrayPortalPermutation(KeyPortalType k,
                         ValuePortalType v):
    Key_(k),
    Value_(v)
    {
    }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->Key_.GetNumberOfValues (); }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const
  {
    return this->Value_.Get(this->Key_.Get(index));
  }

  typedef dax::cont::IteratorFromArrayPortal <ArrayPortalPermutation < KeyPortalType,
                                                                       ValuePortalType> >
  IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
    {
      return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->Key_.GetNumberOfValues ());
  }

private:
  KeyPortalType Key_;
  ValuePortalType Value_;
};

template<class KeyPortalType,class ValuePortalType>
struct ArrayContainerControlTagPermutation {
  typedef ArrayPortalPermutation <KeyPortalType,
                                  ValuePortalType> PortalType;
};

namespace internal {

template<class KeyPortalType,class ValuePortalType>
class ArrayContainerControl<
    typename ValuePortalType::ValueType,
    ArrayContainerControlTagPermutation<KeyPortalType,ValuePortalType> >
{
public:
  typedef typename ValuePortalType::ValueType ValueType;
  typedef ArrayPortalPermutation <KeyPortalType,
                                  ValuePortalType> PortalConstType;

  // This is meant to be invalid. Because Permutation1 arrays are read only, you
  // should only be able to use the const version.
  struct PortalType {
    typedef void *ValueType;
    typedef void *IteratorType;
  };

  // All these methods do nothing but raise errors.
  PortalType GetPortal() {
    throw dax::cont::ErrorControlBadValue("Permutation1 arrays are read-only.");
  }
  PortalConstType GetPortalConst() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "Permutation1 container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  dax::Id GetNumberOfValues() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "Permutation1 container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  void Allocate(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Permutation1 arrays are read-only.");
  }
  void Shrink(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Permutation1 arrays are read-only.");
  }
  void ReleaseResources() {
    throw dax::cont::ErrorControlBadValue("Permutation1 arrays are read-only.");
  }
};

}
}
}

#endif //__dax_cont_ArrayContainerControlPermutation_h
