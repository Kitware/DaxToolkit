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
#ifndef __dax__cont__ArrayContainerControlPermutation1_h
#define __dax__cont__ArrayContainerControlPermutation1_h

#include <dax/Types.h>
#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

namespace dax {
namespace cont {

/// A tag for the Permutation implementation of an ArrayContainerControl object.
struct ArrayContainerControlTagPermutation {  };

namespace internal {

/// A Permutation implementation of an ArrayContainerControl object.
///
/// \todo This container does \em not construct the values within the array.
/// Thus, it is important to not use this class with any type that will fail if
/// not constructed. These are things like basic types (int, float, etc.) and
/// the Dax Tuple classes.  In the future it would be nice to have a compile
/// time check to enforce this.
///
template <typename ValueT>
class ArrayContainerControl<ValueT, dax::cont::ArrayContainerControlTagPermutation>
{
public:
  typedef ValueT ValueType;
  typedef dax::cont::ArrayPortalFromIterators<ValueType*> PortalType;
  typedef dax::cont::ArrayPortalFromIterators<const ValueType*> PortalConstType;

private:
  /// The original design of this class provided an allocator as a template
  /// parameters. That messed things up, though, because other templated
  /// classes assume that the \c ArrayContainerControl has one template
  /// parameter. There are other ways to allow you to specify the allocator,
  /// but it is uncertain whether that would ever be useful. So, instead of
  /// jumping through hoops implementing them, just fix the allocator for now.
  ///
  typedef std::allocator<ValueType> AllocatorType;
  typedef std::allocator<dax::Id> KeyAllocatorType;

public:

  ArrayContainerControl() : Array(NULL), NumberOfValues(0), AllocatedSize(0) { }

  ~ArrayContainerControl()
  {
    this->ReleaseResources();
  }

  void ReleaseResources()
  {
    if (this->NumberOfValues > 0)
      {
      DAX_ASSERT_CONT(this->Array != NULL);
      KeyAllocatorType keyAllocator;
      AllocatorType allocator;
      keyAllocator.deallocate (this->KeyArray, this->NumberOfValues);
      allocator.deallocate(this->Array, this->AllocatedSize);
      this->KeyArray = NULL;
      this->Array = NULL;
      this->NumberOfValues = 0;
      this->AllocatedSize = 0;
      }
    else
      {
      DAX_ASSERT_CONT(this->Array == NULL);
      }
  }

  void Allocate(dax::Id numberOfValues)
  {
    if (numberOfValues <= this->AllocatedSize)
      {
      this->NumberOfValues = numberOfValues;
      return;
      }

    this->ReleaseResources();
    try
      {
      if (numberOfValues > 0)
        {
        KeyAllocatorType keyAllocator;
        AllocatorType allocator;
        this->KeyArray = keyAllocator.allocate (numberOfValues);
        this->Array = allocator.allocate(numberOfValues);
        this->AllocatedSize  = numberOfValues;
        this->NumberOfValues = numberOfValues;
        }
      else
        {
        // ReleaseResources should have already set AllocatedSize to 0.
        DAX_ASSERT_CONT(this->AllocatedSize == 0);
        }
      }
    catch (std::bad_alloc err)
      {
      // Make sureour state is OK.
      this->Array = NULL;
      this->NumberOfValues = 0;
      this->AllocatedSize = 0;
      throw dax::cont::ErrorControlOutOfMemory(
            "Could not allocate Permutation control array.");
      }
  }

  dax::Id GetNumberOfValues() const
  {
    return this->NumberOfValues;
  }

  void Shrink(dax::Id numberOfValues)
  {
    if (numberOfValues > this->GetNumberOfValues())
      {
      throw dax::cont::ErrorControlBadValue(
            "Shrink method cannot be used to grow array.");
      }

    this->NumberOfValues = numberOfValues;
  }

  PortalType GetPortal()
  {
    return PortalType(this->Array, this->Array + this->NumberOfValues);
  }

  PortalConstType GetPortalConst() const
  {
    return PortalConstType(this->Array, this->Array + this->NumberOfValues);
  }

  /// \brief Take the reference away from this object.
  ///
  /// This method returns the pointer to the array held by this array. It then
  /// clears the internal array pointer to NULL, thereby ensuring that the
  /// ArrayContainerControl will never deallocate the array. This is
  /// helpful for taking a reference for an array created internally by Dax and
  /// not having to keep a Dax object around. Obviously the caller becomes
  /// responsible for destroying the memory.
  ///
  ValueType *StealArray()
  {
    ValueType *saveArray =  this->Array;
    this->Array = NULL;
    this->NumberOfValues = 0;
    this->AllocatedSize = 0;
    return saveArray;
  }

private:
  // Not implemented.
  ArrayContainerControl(const ArrayContainerControl<ValueType, ArrayContainerControlTagPermutation> &src);
  void operator=(const ArrayContainerControl<ValueType, ArrayContainerControlTagPermutation> &src);

  dax::Id *KeyArray;
  ValueType *Array;
  dax::Id NumberOfValues;
  dax::Id AllocatedSize;
};

} // namespace internal

}
} // namespace dax::cont

#endif //__dax__cont__ArrayContainerControlPermutation1_h
