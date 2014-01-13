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
#ifndef __dax_cont_ArrayHandleImplicit_h
#define __dax_cont_ArrayHandleImplicit_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {


namespace internal {
/// \brief An array portal that returns the result of a functor
///
/// This array portal is similar to an implicit array i.e an array that is
/// defined functionally rather than actually stored in memory. The array
/// comprises of a functor that is called for each index.
///
/// The ArrayPortalImplicit is used in an ArrayHandle with an
/// ArrayContainerControlTagImplicit container.
///
template <class ValueType_, class FunctorType_ >
class ArrayPortalImplicit
{
public:
  typedef ValueType_ ValueType;
  typedef FunctorType_ FunctorType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalImplicit() :
    Functor(),
    NumberOfValues(0) {  }

  DAX_EXEC_CONT_EXPORT
  ArrayPortalImplicit(FunctorType f, dax::Id numValues) :
    Functor(f),
    NumberOfValues(numValues)
  {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->NumberOfValues; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const { return this->Functor(index); }

  typedef dax::cont::internal::IteratorFromArrayPortal < ArrayPortalImplicit
                                               < ValueType, FunctorType  > >
    IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->NumberOfValues);
  }

private:
  FunctorType Functor;
  dax::Id NumberOfValues;
};

/// A convenience class that provides a typedef to the appropriate tag for
/// a implicit array container.
template<typename ValueType, typename FunctorType>
struct ArrayHandleImplicitTraits
{
  typedef dax::cont::ArrayContainerControlTagImplicit<
      dax::cont::internal::ArrayPortalImplicit<ValueType,
                                               FunctorType> > Tag;
};

} // namespace internal


/// ArrayHandleImplicits is a specialization of ArrayHandle.
/// It takes a user defined functor which is called with a given index value.
/// The functor returns the result of the functor as the value of this
/// array at that position.
///

template <typename ValueType,
          class FunctorType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleImplicit
    : public dax::cont::ArrayHandle <
          ValueType,
          typename internal::ArrayHandleImplicitTraits<ValueType,
                                                       FunctorType>::Tag,
          DeviceAdapterTag >
{
private:
  typedef typename internal::ArrayHandleImplicitTraits<ValueType,
                                FunctorType> ArrayTraits;

  typedef typename ArrayTraits::Tag Tag;

 public:
  typedef dax::cont::ArrayHandle < ValueType,
                                   Tag,
                                   DeviceAdapterTag > Superclass;

  ArrayHandleImplicit(): Superclass() {}

  ArrayHandleImplicit(FunctorType functor, dax::Id length)
      :Superclass(typename Superclass::PortalConstControl(functor,length))
    {
    }
};

/// make_ArrayHandleImplicit is convenience function to generate an
/// ArrayHandleImplicit.  It takes a functor and the virtual length of the
/// arry.

template <typename T, typename FunctorType>
DAX_CONT_EXPORT
dax::cont::ArrayHandleImplicit<T, FunctorType>
make_ArrayHandleImplicit(FunctorType functor, dax::Id length)
{
  return ArrayHandleImplicit<T,FunctorType>(functor,length);
}


}
} // namespace dax::cont

#endif //__dax_cont_ArrayHandleImplicit_h
