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

#ifndef __dax_cont_PermutationContainer_h
#define __dax_cont_PermutationContainer_h

#include <dax/Types.h>

namespace dax {
namespace cont {

/// \headerfile PermutationContainer.h dax/cont/PermutationContainer.h
/// \brief  Creates a Key Value pair of objects that will be used
/// as a single parameter on the execution side for a worklet.

/// The presumption currently is that the Key object has a length
/// that is equal to the numbers of elements we are scheduling. We
/// will create the ConceptMap for Key and ask it to create it's size
/// to that number.
/// The presumption currently is that the Value object has a length
/// that is unkown, so when uploading to the execution enviornment, we
/// will call GetNumberOfValues() on it and pass that value to the
/// ConceptMap of the Value Object as the size to upload ( ToExecution method ).
/// If the object doesn't have a size since it self, you can set
/// the SetValueSize method on the PermutationContainer to give a custom allocation
/// size

template<class KeyType,
         class ValueType>
class PermutationContainer
{
public:
  DAX_CONT_EXPORT PermutationContainer(KeyType& k,
                                       ValueType& v,
                                       dax::Id valueLen):
    Key_(k),
    Value_(v),
    ValueSizeToAllocate(valueLen)
  {
  }

  DAX_CONT_EXPORT void SetValueSize(dax::Id v) { this->ValueSizeToAllocate = v; }
  DAX_CONT_EXPORT dax::Id GetValueSize() const
  {
    return this->ValueSizeToAllocate;
  }

  //should really only be used by the FieldMap Concept
  DAX_CONT_EXPORT KeyType Key() const { return Key_; }
  DAX_CONT_EXPORT ValueType Value() const { return Value_; }

private:

  KeyType& Key_;
  ValueType& Value_;
  dax::Id ValueSizeToAllocate;
};


template<class Key, class Value>
DAX_CONT_EXPORT
dax::cont::PermutationContainer<Key,Value>
make_Permutation(Key& k,Value& v, dax::Id valueLen)
{
  return dax::cont::PermutationContainer<Key,Value>(k,v,valueLen);
}

} } // namespace dax::cont

#endif // __dax_cont_PermutationContainer_h
