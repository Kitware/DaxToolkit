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
#ifndef __dax__cont__ArrayContainerControl_h
#define __dax__cont__ArrayContainerControl_h

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY
/// \brief A tag specifying client memory allocation.
///
/// An ArrayContainerControl tag specifies how an ArrayHandle allocates and
/// frees memory. The tag ArrayContainerControlTag___ does not actually exist.
/// Rather, this documentation is provided to describe how array containers are
/// specified. Loading the dax/cont/ArrayContainerControl.h header will set a
/// default array container. The default array container can be overloaded by
/// including the header file for a differt array container (for example,
/// ArrayContainerControlBasic.h) or simply defining the macro
/// DAX_DEFAULT_ARRAY_CONTAINER_CONTROL. This overloading should be done \em
/// before loading in any other Dax header files (with the exception of a
/// DeviceAdapter header file). Failing to do so could create inconsistencies
/// in the default adapter used among classes.
///
/// User code external to Dax is free to make its own ArrayContainerControlTag.
/// This is a good way to get Dax to read data directly in and out of arrays
/// from other libraries. However, care should be taken when creating an
/// ArrayContainerControl. One particular problem that is likely is a container
/// that "constructs" all the items in the array. If done incorrectly, then
/// memory of the array can be incorrectly bound to the wrong process. If you
/// do provide your own ArrayContainerControlTag, please be diligent in
/// comparing its performance to the ArrayContainerControlTagBasic.
///
/// To implement your own ArrayContainerControlTag, you first must create a tag
/// class (an empty struct) defining your tag (i.e. struct
/// ArrayContainerControlTagMyAlloc { };). Then provide a partial template
/// specialization of dax::cont::internal::ArrayContainerControl for your new
/// tag.
///
struct ArrayContainerControlTag___ {  };
#endif // DAX_DOXYGEN_ONLY

namespace internal {

/// This templated class must be partially specialized for each
/// ArrayContainerControlTag created, which will define the implementation for
/// that tag.
///
template<typename T, class ArrayContainerControlTag>
class ArrayContainerControl
#ifdef DAX_DOXYGEN_ONLY
{
public:

  /// The type of each item in the array.
  ///
  typedef T ValueType;

  /// \brief The type of iterator objects for the array.
  ///
  /// The actual iterator object may be more complicated, much like the
  /// iterators in STL containers.
  ///
  typedef ValueType *IteratorType;

  /// \brief The type of iterator objects (const version) for the array.
  ///
  /// The actual iterator object may be more complicated, much like the
  /// iterators in STL containers.
  ///
  typedef const ValueType *IteratorConstType;

  /// Returns the iterator at the beginning of the array.
  ///
  IteratorType GetIteratorBegin();

  /// Returns the iterator at the end of the array.
  ///
  IteratorType GetIteratorEnd();

  /// Returns the const iterator at the beginning of the array.
  ///
  IteratorConstType GetIteratorConstBegin() const;

  /// Returns the const iterator at the end of the array.
  ///
  IteratorConstType GetIteratorConstEnd() const;

  /// Retuns the number of entries allocated in the array.
  dax::Id GetNumberOfValues() const;

  /// \brief Allocates an array large enough to hold the given number of values.
  ///
  /// The allocation may be done on an already existing array, but can wipe out
  /// any data already in the array. This method can throw
  /// ErrorControlOutOfMemory if the array cannot be allocated.
  ///
  void Allocate(dax::Id numberOfValues);

  /// \brief Reduces the size of the array without changing its values.
  ///
  /// This method allows you to resize the array without reallocating it. The
  /// number of entries in the array is changed to \c numberOfValues. The data
  /// in the array (from indices 0 to \c numberOfValues - 1) are the same, but
  /// \c numberOfValues must be equal or less than the preexisting size
  /// (returned from GetNumberOfValues). That is, this method can only be used
  /// to shorten the array, not lengthen.
  void Shrink(dax::Id numberOfValues);

  /// \brief Frees any resources (i.e. memory) stored in this array.
  ///
  /// After calling this method GetNumberOfValues will return 0 and
  /// GetIteratorBegin and GetIteratorEnd will return the same iterator. The
  /// resources should also be released when the ArrayContainerControl class is
  /// destroyed.
  void ReleaseResources();
};
#else // DAX_DOXYGEN_ONLY
;
#endif // DAX_DOXYGEN_ONLY

} // namespace internal

}
} // namespace dax::cont

// This is put at the bottom of the header so that the ArrayContainerControl
// template is declared before any implementations are called.
#ifndef DAX_DEFAULT_ARRAY_CONTAINER_CONTROL
#include <dax/cont/ArrayContainerControlBasic.h>
#endif // DAX_DEFAULT_ARRAY_CONTAINER_CONTROL

#endif //__dax__cont__ArrayContainerControl_h
