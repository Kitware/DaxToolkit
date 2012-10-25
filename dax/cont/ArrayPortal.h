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
#ifndef __dax_cont_ArrayPortal_h
#define __dax_cont_ArrayPortal_h

#include <dax/Types.h>

namespace dax {
namespace cont {

#ifdef DAX_DOXYGEN_ONLY

/// \brief A class that points to and access and array of data.
///
/// The ArrayPortal class itself does not exist; this code is provided for
/// documentation purposes only.
///
/// An ArrayPortal object acts like a pointer to a random-access container
/// (that is, an array) and also lets you set and get values in that array. In
/// many respects an ArrayPortal is similar in concept to that of iterators but
/// with a much simpler interface and no internal concept of position.
/// Otherwise, ArrayPortal objects may be passed and copied around so that
/// multiple entities may be accessing the same array.
///
/// An ArrayPortal differs from an ArrayHandle in that the ArrayPortal is a
/// much lighterweight object and that it does not manage things like
/// allocation and control/execution sharing. An ArrayPortal also differs from
/// an ArrayContainer in that it does not actually contain the data but rather
/// points to it. In this way the ArrayPortal can be copied and passed and
/// still point to the same data.
///
/// Most Dax users generally do not need to do much or anything with
/// ArrayPortal objects. It is mostly an internal mechanism. However, an
/// ArrayPortal can be used to pass constant input data to an ArrayHandle.
///
/// Although this documentation is given for the control environment, there are
/// instances of an identical concept in the execution environment, although
/// some features are missing there.
///
template<typename T>
class ArrayPortal
{
public:
  /// The type of each value in the array.
  ///
  typedef T ValueType;

  /// The total number of values in the array. They are index from 0 to
  /// GetNumberOfValues()-1.
  ///
  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const;

  /// Gets a value from the array.
  ///
  DAX_CONT_EXPORT
  ValueType Get(dax::Id index) const;

  /// Sets a value in the array. This function may not exist for an ArrayPortal
  /// pointing to a const array.
  ///
  DAX_CONT_EXPORT
  void Set(dax::Id index, const ValueType &value) const;

  /// An iterator type that can be used as an alternate way to access the data.
  /// If the container being pointed to has a natural iterator that can be
  /// used, then use that. Otherwise, use IteratorForArrayPortal. Iterators are
  /// not necessary for array portals in the execution environment.
  ///
  typedef ValueType *IteratorType;

  /// Returns an iterator to the beginning of the array. Iterators are not
  /// necessary for array portals in the execution environment.
  ///
  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const;

  /// Returns an iterator to the end of the array. Iterators are not necessary
  /// for array portals in the execution environment.
  ///
  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const;
};

#endif // DAX_DOXYGEN_ONLY

}
} // namespace dax::cont

#endif //__dax_cont_ArrayPortal_h
