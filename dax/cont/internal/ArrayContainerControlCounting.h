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
#ifndef __dax_cont_internal_ArrayContainerControlCounting_h
#define __dax_cont_internal_ArrayContainerControlCounting_h

#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {
namespace internal{

/// \brief An implicit array portal that returns an index.
///
/// This array portal points to an implicit array (that is, an array that is
/// defined functionally rather than actually stored in memory). The array
/// comprises consecutive indices. That is, the values [0, 1, 2, 3,...].
///
/// The ArrayPortalCounting is used in an ArrayHandle with an
/// ArrayContainerControlTagCounting container.
///
class ArrayPortalCounting
{
public:
  typedef dax::Id ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting() : LastIndex(0) {  }

  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting(dax::Id numValues) : LastIndex(numValues) {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->LastIndex; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const { return index; }

  typedef dax::cont::internal::IteratorFromArrayPortal<ArrayPortalCounting> IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->LastIndex);
  }

private:
  // The current implementation of this class only allows you to specify the
  // number of indices and then gives indices 0..(size-1). This is the most
  // common case although there may be uses for offset indices (starting at
  // something other than 0) or different strides. These are easy enough to
  // implement, but I have not because I currently have no need for them and
  // they require a bit more state (that must be copied around). If these extra
  // features are needed, they can be added in the future.

  // The last index given by this portal (exclusive).
  dax::Id LastIndex;
};

/// \brief An implicit array storing consecutive indices.
///
/// This array container points to an implicit array (that is, an array that is
/// defined functionally rather than actually stored in memory). The array
/// comprises consecutive indices. That is, the values [0, 1, 2, 3,...].
///
/// When creating an ArrayHandle with an ArrayContainerControlTagImplicit
/// container, use an ArrayPortalCounting to establish the array.
///
typedef ArrayContainerControlTagImplicit<dax::cont::internal::ArrayPortalCounting>
    ArrayContainerControlTagCounting;

}
}
}

#endif //__dax_cont_internal_ArrayContainerControlCounting_h
