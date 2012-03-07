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
#ifndef __dax_cont_internal_IteratorContainer_h
#define __dax_cont_internal_IteratorContainer_h

#include <dax/Types.h>
#include <dax/cont/Assert.h>

#include <iterator>

namespace dax {
namespace cont {
namespace internal {

/// A simple container class for a pair of begin/end iterators. This class
/// provides a mechanism for holding a reference to an array (or something that
/// looks like an array) without having to specify any container.
///
template<typename IterT>
class IteratorContainer
{
public:
  typedef IterT IteratorType;
  typedef typename std::iterator_traits<IteratorType>::value_type ValueType;

  IteratorContainer() : Valid(false) { }
  IteratorContainer(IteratorType begin, IteratorType end)
    : BeginIterator(begin), EndIterator(end), Valid(true) { }

  /// Returns true if the iterators are valid.  This is not actually checked
  /// (there is no way to do so).  Rather, the iterators are considered valid
  /// until Invalidate is called.
  ///
  bool IsValid() const { return this->Valid; }

  /// Call this method if the iterators become invalid (such as if the memory
  /// is freed).
  ///
  void Invalidate() { this->Valid = false; }

  /// Returns the begin iterator. Behavior is undefined when IsValid() returns
  /// false.
  ///
  IteratorType GetBeginIterator() const {
    DAX_ASSERT_CONT(this->Valid);
    return this->BeginIterator;
  }

  /// Returns the end iterator. Behavior is undefined when IsValid() returns
  /// false.
  ///
  IteratorType GetEndIterator() const {
    DAX_ASSERT_CONT(this->Valid);
    return this->EndIterator;
  }

  /// Returns the number of entries between the begin and end iterators.
  /// Behavior is undefined when IsValid() returns false.
  ///
  dax::Id GetNumberOfEntries() const {
    return this->GetEndIterator() - this->GetBeginIterator();
  }

private:
  IteratorType BeginIterator;
  IteratorType EndIterator;
  bool Valid;
};

/// A convenience function to build an interator container. Helpful to
/// automatically type the necessary function.
///
template<class IteratorType>
IteratorContainer<IteratorType> make_IteratorContainer(IteratorType begin,
                                                       IteratorType end)
{
  return IteratorContainer<IteratorType>(begin, end);
}

/// A convenience function to build an interator container. Helpful to
/// automatically type the necessary function. Also helpful when you have the
/// begin random iterator and a size.
///
template<class IteratorType>
IteratorContainer<IteratorType> make_IteratorContainer(IteratorType begin,
                                                       dax::Id size)
{
  IteratorType end = begin;
  std::advance(end, size);
  return IteratorContainer<IteratorType>(begin, end);
}

}
}
}

#endif //__dax_cont_internal_IteratorContainer_h
