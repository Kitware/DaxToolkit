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
#ifndef __dax__cont__ArrayContainerControlBasic_h
#define __dax__cont__ArrayContainerControlBasic_h

#ifdef DAX_DEFAULT_ARRAY_CONTAINER_CONTROL
#undef DAX_DEFAULT_ARRAY_CONTAINER_CONTROL
#endif

#define DAX_DEFAULT_ARRAY_CONTAINER_CONTROL ::dax::cont::ArrayContainerControlBasic

#include <dax/Types.h>
#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

namespace dax {
namespace cont {

/// A basic implementation of an ArrayContainerControl object.
///
/// \todo This container does \em not construct the values within the array.
/// Thus, it is important to not use this class with any type that will fail if
/// not constructed. These are things like basic types (int, float, etc.) and
/// the Dax Tuple classes.
///
template <typename ValueT, template <typename> class Allocator = std::allocator>
class ArrayContainerControlBasic
{
public:
  typedef ValueT ValueType;
  typedef ValueType *IteratorType;

  ArrayContainerControlBasic() : Internals(new SharedInternals)
  {
    this->Internals->Array = NULL;
    this->Internals->NumberOfValues = 0;
    this->Internals->ReferenceCount = 1;
  }
  ArrayContainerControlBasic(
      const ArrayContainerControlBasic<ValueType,Allocator> &src)
    : Internals(src.Internals)
  {
    this->Internals->ReferenceCount++;
  }
  ArrayContainerControlBasic<ValueType,Allocator> &operator=(
      const ArrayContainerControlBasic<ValueType,Allocator> &src)
  {
    if (this->Internals == src.Internals) return *this;
    this->ReleaseReference();
    this->Internals = src.Internals;
    this->Internals->ReferenceCount++;
    return *this;
  }

  ~ArrayContainerControlBasic()
  {
    this->ReleaseReference();
  }

  void ReleaseResources()
  {
    if (this->Internals->NumberOfValues > 0)
      {
      DAX_ASSERT_CONT(this->Internals->Array != NULL);
      Allocator<ValueType> allocator;
      allocator.deallocate(this->Internals->Array,
                           this->Internals->NumberOfValues);
      this->Internals->Array = NULL;
      this->Internals->NumberOfValues = 0;
      }
    else
      {
      DAX_ASSERT_CONT(this->Internals->Array == NULL);
      }
  }

  void Allocate(dax::Id numberOfValues)
  {
    if (this->Internals->NumberOfValues == numberOfValues) return;

    this->ReleaseResources();
    try
      {
      if (numberOfValues > 0)
        {
        Allocator<ValueType> allocator;
        this->Internals->Array = allocator.allocate(numberOfValues);
        this->Internals->NumberOfValues = numberOfValues;
        }
      else
        {
        // ReleaseResources should have already set NumberOfValues to 0.
        }
      }
    catch (std::bad_alloc err)
      {
      // Make sureour state is OK.
      this->Internals->Array = NULL;
      this->Internals->NumberOfValues = 0;
      throw dax::cont::ErrorControlOutOfMemory(
            "Could not allocate basic control array.");
      }
  }

  dax::Id GetNumberOfValues() const
  {
    return this->Internals->NumberOfValues;
  }

  IteratorType GetIteratorBegin() const
  {
    return this->Internals->Array;
  }

  IteratorType GetIteratorEnd() const
  {
    return this->GetIteratorBegin() + this->GetNumberOfValues();
  }

  /// \brief Take the reference away from this object.
  ///
  /// This method returns the pointer to the array held by this array (and any
  /// copies). It then clears the internal array pointer to NULL, thereby
  /// ensuring that the ArrayContainerControlBasic will never deallocate the
  /// array. This is helpful for taking a reference for an array created
  /// internally by Dax and not having to keep a Dax object around. Obviously
  /// the caller becomes responsible for destroying the memory.
  ///
  ValueType *StealArray()
  {
    ValueType *saveArray =  this->Internals->Array;
    this->Internals->Array = NULL;
    this->Internals->NumberOfValues = 0;
    return saveArray;
  }

private:
  struct SharedInternals
  {
    ValueType *Array;
    dax::Id NumberOfValues;
    dax::Id ReferenceCount;
  };

  SharedInternals *Internals;

  /// Should only be called if this->Internals will never be used again.
  void ReleaseReference()
  {
    DAX_ASSERT_CONT(this->Internals->ReferenceCount > 0);
    this->Internals->ReferenceCount--;
    if (this->Internals->ReferenceCount == 0)
      {
      this->ReleaseResources();
      delete this->Internals;
      }
    else
      {
      // Other copies still using array.  Leave alone.
      }
    this->Internals = NULL;
  }
};

}
} // namespace dax::cont

#endif //__dax__cont__ArrayContainerControlBasic_h
