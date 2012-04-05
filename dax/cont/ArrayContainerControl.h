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

#ifndef DAX_DEFAULT_ARRAY_CONTAINER_CONTROL
// Include for basic container.
#endif // DAX_DEFAULT_ARRAY_CONTAINER_CONTROL

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY

/// \brief Abstract interface to client memory allocation.
///
/// An ArrayContainerControl is a class that behaves like a container class
/// (sort of like std::vector but more basic). The class
/// dax::cont::ArrayContainerControl does not actually exist. Rather, this
/// documentation is provided to describe the interface for an
/// ArrayContainerControl. Loading the dax/cont/ArrayContainerControl.h header
/// will set a default array container. The default array container can be
/// overloaded by including the header file for a differt array container (for
/// example, ArrayContainerControlBasic.h) or simply defining the macro
/// DAX_DEFAULT_ARRAY_CONTAINER_CONTROL. This overloading should be done
/// \em before loading in any other Dax header files (with the exception of a
/// DeviceAdapter header file). Failing to do so could create inconsistencies
/// in the default adapter used among classes.
///
/// User code external to Dax is free to make its own ArrayContainerControl
/// class. This is a good way to get Dax to read data directly in and out of
/// arrays from other libraries. However, care should be taken when creating an
/// ArrayContainerControl. One particular problem that is likely is a container
/// that "constructs" all the items in the array. If done incorrectly, then
/// memory of the array can be incorrectly bound to the wrong process.  If you
/// do provide your own ArrayContainerControl, please be diligent in comparing
/// its performance to the ArrayContainerControlBasic.
///
/// In addition to the performance constraints listed above, an implementation
/// of ArrayContainerControl is expected to exhibit the following behaviors (in
/// addition to the obvious documentation of members).
///
/// \param A copy of an ArrayContainerControl (either through a copy
/// constructor or a copy operator) should do a shallow copy.
///
/// \param An Allocate or ReleaseResources on any copy of the array should
/// change the memory reference in all copies of the ArrayContainerControl.
///
/// \param The ArrayContainerControl should internally reference count the
/// number of copies pointing to memory and automatically free the memory when
/// the count reaches 0.
///
/// These requirements together basically state that the ArrayContainerControl
/// should be have like a reference counted smart pointer.
///
template <typename ValueT>
class ArrayContainerControl
{
public:

  /// The type of each item in the array.
  ///
  typedef ValueT ValueType;

  /// \brief The type of iterator objects for the array.
  ///
  /// The actual iterator object may be more complicated, much like the
  /// iterators in STL containers.
  ///
  typedef ValueType *IteratorType;

  /// Returns the iterator at the beginning of the array.
  ///
  IteratorType GetIteratorBegin() const;

  /// Returns the iterator at the end of the array.
  ///
  IteratorType GetIteratorEnd() const;

  /// Retuns the number of entries allocated in the array.
  dax::Id GetNumberOfValues() const;

  /// \brief Allocates an array large enough to hold the given number of values.
  ///
  /// The allocation may be done on an already existing array, but can wipe out
  /// any data already in the array. This method can throw
  /// ErrorControlOutOfMemory if the array cannot be allocated.
  ///
  void Allocate(dax::Id numberOfValues);

  /// \brief Frees any resources (i.e. memory) stored in this array.
  ///
  /// After calling this method GetNumberOfValues will return 0 and
  /// GetIteratorBegin and GetIteratorEnd will return the same iterator. The
  /// resources should also be released when the ArrayContainerControl class is
  /// destroyed.
  void ReleaseResources();
};

#endif

}
} // namespace dax::cont

#endif //__dax__cont__ArrayContainerControl_h
