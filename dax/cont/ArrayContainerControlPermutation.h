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

/// \brief An implicit array portal that returns a permuted value.
///
/// This array portal points to an implicit array (perhaps it should not be but
/// for the time being it is). The array comprises of two arrays one with the
/// Key and other with the Value. The Key holds the index into Value array and
/// returns the corresponding value. So for example if the Key array = [0,2,3,5]
/// and Value array = [8,6,4,9,8,3]. The array[2] = Value[Key[2]] =
/// Value[3] = 9.
///
/// The ArrayPortalPermutation is used in an ArrayHandle with an
/// ArrayContainerControlTagPermutation container.
///
class ArrayPortalPermutation
{
public:
  typedef dax::Id ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalPermutation(): KeyLength(0), ValueLength(0) {}

  DAX_EXEC_CONT_EXPORT ArrayPortalPermutation(dax::Id* k,
                                              dax::Id keyLen,
                                              ValueType* v,
                                              dax::Id valueLen):
    Key_(k),
    Value_(v),
    KeyLength(keyLen),
    ValueLength(valueLen)
    {
      for (dax::Id i = 0; i < keyLen; ++i)
        {
        DAX_ASSERT_CONT(k[i] < valueLen);
        }
    }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->KeyLength; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const
  {
    return this->Value_[this->Key_[index]];
  }

  typedef dax::cont::IteratorFromArrayPortal<ArrayPortalPermutation> IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
    {
      return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->KeyLength);
  }

private:
  dax::Id* Key_;
  ValueType* Value_;
  dax::Id KeyLength;
  dax::Id ValueLength;
};

/// \brief An implicit array storing consecutive indices.
///
/// This array portal points to an implicit array (perhaps it should not be but
/// for the time being it is). The array comprises of two arrays one with the
/// Key and other with the Value. The Key holds the index into Value array and
/// returns the corresponding value. So for example if the Key array = [0,2,3,5]
/// and Value array = [8,6,4,9,8,3]. The array[2] = Value[Key[2]] =
/// Value[3] = 9.
///
/// When creating an ArrayHandle with an ArrayContainerControlTagImplicit
/// container, use an ArrayPortalPermutation to establish the array.
///
typedef ArrayContainerControlTagImplicit<dax::cont::ArrayPortalPermutation>
    ArrayContainerControlTagPermutation;

}
}

#endif //__dax_cont_ArrayContainerControlPermutation_h
