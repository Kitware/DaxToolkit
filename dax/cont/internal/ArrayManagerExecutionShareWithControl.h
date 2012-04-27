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

#include <algorithm>

namespace dax {
namespace cont {
namespace internal {

/// \c ArrayManagerExecutionShareWithControl provides an implementation for a
/// \c ArrayManagerExecution sublcass of a \c DeviceAdapter when the execution
/// and control environments share memory. This class basically defers all its
/// calls to an \c ArrayContainerControl class and uses the array allocated
/// there.
///
template<typename T, template <typename> class ArrayContainerControl>
class ArrayManagerExecutionShareWithControl
{
public:
  typedef T ValueType;
  typedef typename ArrayContainerControl<ValueType>::IteratorType IteratorType;

  DAX_CONT_EXPORT ArrayManagerExecutionShareWithControl()
    : IteratorsValid(false) { }

  /// Saves the given iterators to be returned later.
  ///
  DAX_CONT_EXPORT void LoadDataForInput(IteratorType beginIterator,
                                        IteratorType endIterator) {
    this->BeginIterator = beginIterator;
    this->EndIterator = endIterator;
    this->IteratorsValid = true;
  }

  /// Actually just allocates memory in the given \p controlArray.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(
      ArrayContainerControl<ValueType> &controlArray,
      dax::Id numberOfValues)
  {
    controlArray.Allocate(numberOfValues);
    this->BeginIterator = controlArray.GetIteratorBegin();
    this->EndIterator = controlArray.GetIteratorEnd();
    this->IteratorsValid = true;
  }

  /// This method is a no-op (except for a few checks). Any data written to
  /// this class's iterators should already be written to the give \c
  /// controlArray (under correct operation).
  ///
  DAX_CONT_EXPORT void RetreiveOutputData(
      ArrayContainerControl<ValueType> &controlArray) const
  {
    DAX_ASSERT_CONT(this->IteratorsValid);
    DAX_ASSERT_CONT(controlArray.GetIteratorBegin() == this->BeginIterator);
    DAX_ASSERT_CONT(controlArray.GetIteratorEnd() == this->EndIterator);
  }

  /// This methods copies data from the execution array into the given
  /// iterator.
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    std::copy(this->BeginIterator, this->EndIterator, dest);
  }

  /// Returns the iterator previously saved from an \c ArrayContainerControl.
  ///
  DAX_CONT_EXPORT IteratorType GetIteratorBegin() const
  {
    DAX_ASSERT_CONT(this->IteratorsValid);
    return this->BeginIterator;
  }

  /// Returns the iterator previously saved from an \c ArrayContainerControl.
  ///
  DAX_CONT_EXPORT IteratorType GetIteratorEnd() const
  {
    DAX_ASSERT_CONT(this->IteratorsValid);
    return this->EndIterator;
  }

  /// A no-op.
  ///
  DAX_CONT_EXPORT void ReleaseResources() { }

private:
  // Not implemented.
  ArrayManagerExecutionShareWithControl(
      ArrayManagerExecutionShareWithControl<T, ArrayContainerControl> &);
  void operator=(
      ArrayManagerExecutionShareWithControl<T, ArrayContainerControl> &);

  IteratorType BeginIterator;
  IteratorType EndIterator;
  bool IteratorsValid;
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayManagerExecutionShareWithControl_h
