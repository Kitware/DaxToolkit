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
#ifndef __dax_cont_internal_ArrayManagerExecutionShareWithControl_h
#define __dax_cont_internal_ArrayManagerExecutionShareWithControl_h

#include <dax/Types.h>

#include <dax/cont/Assert.h>
#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/internal/ArrayPortalShrink.h>

#include <algorithm>

namespace dax {
namespace cont {
namespace internal {

/// \c ArrayManagerExecutionShareWithControl provides an implementation for a
/// \c ArrayManagerExecution class for a device adapter when the execution
/// and control environments share memory. This class basically defers all its
/// calls to an \c ArrayContainerControl class and uses the array allocated
/// there.
///
template<typename T, class ArrayContainerControlTag>
class ArrayManagerExecutionShareWithControl
{
public:
  typedef T ValueType;
  typedef dax::cont::internal
      ::ArrayContainerControl<ValueType, ArrayContainerControlTag>
      ContainerType;
  typedef dax::cont::internal::ArrayPortalShrink<
      typename ContainerType::PortalType> PortalType;
  typedef dax::cont::internal::ArrayPortalShrink<
      typename ContainerType::PortalConstType> PortalConstType;

  DAX_CONT_EXPORT ArrayManagerExecutionShareWithControl()
    : PortalValid(false), ConstPortalValid(false) { }

  /// Saves the given iterators to be returned later.
  ///
  DAX_CONT_EXPORT void LoadDataForInput(PortalConstType portal)
  {
    this->ConstPortal = portal;
    this->ConstPortalValid = true;

    // Non-const versions not defined.
    this->PortalValid = false;
  }

  /// Saves the given iterators to be returned later.
  ///
  DAX_CONT_EXPORT void LoadDataForInput(PortalType portal)
  {
    // This only works if there is a valid cast from non-const to const
    // iterator.
    this->LoadDataForInput(PortalConstType(portal));

    this->Portal = portal;
    this->PortalValid = true;
  }

  /// Actually just allocates memory in the given \p controlArray.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(ContainerType &controlArray,
                                              dax::Id numberOfValues)
  {
    controlArray.Allocate(numberOfValues);

    this->Portal = controlArray.GetPortal();
    this->PortalValid = true;

    this->ConstPortal = controlArray.GetPortalConst();
    this->ConstPortalValid = true;
  }

  /// This method is a no-op (except for a few checks). Any data written to
  /// this class's iterators should already be written to the given \c
  /// controlArray (under correct operation).
  ///
  DAX_CONT_EXPORT void RetrieveOutputData(ContainerType &controlArray) const
  {
    DAX_ASSERT_CONT(this->ConstPortalValid);
    DAX_ASSERT_CONT(controlArray.GetPortalConst().GetIteratorBegin() ==
                    this->ConstPortal.GetIteratorBegin());
  }

  /// This methods copies data from the execution array into the given
  /// iterator.
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    DAX_ASSERT_CONT(this->ConstPortalValid);
    std::copy(this->ConstPortal.GetIteratorBegin(),
              this->ConstPortal.GetIteratorEnd(),
              dest);
  }

  /// Adjusts saved end iterators to resize array.
  ///
  DAX_CONT_EXPORT void Shrink(dax::Id numberOfValues)
  {
    DAX_ASSERT_CONT(this->ConstPortalValid);
    this->ConstPortal.Shrink(numberOfValues);

    if (this->PortalValid)
      {
      this->Portal.Shrink(numberOfValues);
      }
  }

  /// Returns the portal previously saved from an \c ArrayContainerControl.
  ///
  DAX_CONT_EXPORT PortalType GetPortal()
  {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->Portal;
  }

  /// Const version of GetPortal.
  ///
  DAX_CONT_EXPORT PortalConstType GetPortalConst() const
  {
    DAX_ASSERT_CONT(this->ConstPortalValid);
    return this->ConstPortal;
  }

  /// A no-op.
  ///
  DAX_CONT_EXPORT void ReleaseResources() { }

private:
  // Not implemented.
  ArrayManagerExecutionShareWithControl(
      ArrayManagerExecutionShareWithControl<T, ArrayContainerControlTag> &);
  void operator=(
      ArrayManagerExecutionShareWithControl<T, ArrayContainerControlTag> &);

  PortalType Portal;
  bool PortalValid;

  PortalConstType ConstPortal;
  bool ConstPortalValid;
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayManagerExecutionShareWithControl_h
