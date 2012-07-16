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

#define DAX_DEFAULT_ARRAY_CONTAINER_CONTROL \
  ::dax::cont::ArrayContainerControlTagBasic

#include <dax/Types.h>
#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

namespace dax {
namespace cont {

/// A tag for the basic implementation of an ArrayContainerControl object.
struct ArrayContainerControlTagBasic {  };

namespace internal {

/// A basic implementation of an ArrayContainerControl object.
///
/// \todo This container does \em not construct the values within the array.
/// Thus, it is important to not use this class with any type that will fail if
/// not constructed. These are things like basic types (int, float, etc.) and
/// the Dax Tuple classes.  In the future it would be nice to have a compile
/// time check to enforce this.
///
template <typename ValueT>
class ArrayContainerControl<ValueT, dax::cont::ArrayContainerControlTagBasic>
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

public:

  ArrayContainerControl() : Array(NULL), NumberOfValues(0) { }

  ~ArrayContainerControl()
  {
    this->ReleaseResources();
  }

  void ReleaseResources()
  {
    if (this->NumberOfValues > 0)
      {
      DAX_ASSERT_CONT(this->Array != NULL);
      AllocatorType allocator;
      allocator.deallocate(this->Array, this->NumberOfValues);
      this->Array = NULL;
      this->NumberOfValues = 0;
      }
    else
      {
      DAX_ASSERT_CONT(this->Array == NULL);
      }
  }

  void Allocate(dax::Id numberOfValues)
  {
    if (this->NumberOfValues == numberOfValues) return;

    this->ReleaseResources();
    try
      {
      if (numberOfValues > 0)
        {
        AllocatorType allocator;
        this->Array = allocator.allocate(numberOfValues);
        this->NumberOfValues = numberOfValues;
        }
      else
        {
        // ReleaseResources should have already set NumberOfValues to 0.
        DAX_ASSERT_CONT(this->NumberOfValues == 0);
        }
      }
    catch (std::bad_alloc err)
      {
      // Make sureour state is OK.
      this->Array = NULL;
      this->NumberOfValues = 0;
      throw dax::cont::ErrorControlOutOfMemory(
            "Could not allocate basic control array.");
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
    return saveArray;
  }

private:
  // Not implemented.
  ArrayContainerControl(const ArrayContainerControl<ValueType, ArrayContainerControlTagBasic> &src);
  void operator=(const ArrayContainerControl<ValueType, ArrayContainerControlTagBasic> &src);

  ValueType *Array;
  dax::Id NumberOfValues;
};

} // namespace internal

}
} // namespace dax::cont

#endif //__dax__cont__ArrayContainerControlBasic_h
