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
#ifndef __dax_cont_internal_ArrayHandleTransform_h
#define __dax_cont_internal_ArrayHandleTransform_h

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ErrorControlInternal.h>
#include <dax/cont/internal/ArrayTransfer.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {
namespace internal {

/// \brief An implicit array portal that returns an counting value.
template <class ValueType_,
          class PortalType_,
          class FunctorType_>
class ArrayPortalTransform
{
public:
  typedef PortalType_ PortalType;
  typedef ValueType_ ValueType;
  typedef FunctorType_ FunctorType;

  DAX_CONT_EXPORT
  ArrayPortalTransform() :
  Portal(),
  NumberOfValues(0),
  Functor()
  {  }

  DAX_CONT_EXPORT
  ArrayPortalTransform(const PortalType& portal, dax::Id size, FunctorType f ) :
  Portal(portal),
  NumberOfValues(size),
  Functor(f)
  {  }

  /// Copy constructor for any other ArrayPortalTransform with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<class OtherV, class OtherP, class OtherF>
  DAX_CONT_EXPORT
  ArrayPortalTransform(const ArrayPortalTransform<OtherV,OtherP,OtherF> &src)
    : Portal(src.GetPortal()),
      NumberOfValues(src.GetNumberOfValues()),
      Functor(src.GetFunctor())

  {  }

  DAX_EXEC_EXPORT
  dax::Id GetNumberOfValues() const {
    return this->NumberOfValues;
  }

  DAX_EXEC_EXPORT
  ValueType Get(dax::Id index) const{
    return this->Functor(this->Portal.Get(index));
  }

  typedef dax::cont::internal::IteratorFromArrayPortal< ArrayPortalTransform <
                    ValueType, PortalType, FunctorType> > IteratorType;

  DAX_EXEC_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_EXEC_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->GetNumberOfValues());
  }

  DAX_CONT_EXPORT
  PortalType &GetPortal() { return this->Portal; }
  DAX_CONT_EXPORT
  const PortalType &GetPortal() const { return this->Portal; }

  DAX_CONT_EXPORT
  FunctorType &GetFunctor() { return this->Portal; }
  DAX_CONT_EXPORT
  const FunctorType &GetFunctor() const { return this->Portal; }

private:
  PortalType Portal;
  dax::Id NumberOfValues;
  FunctorType Functor;
};

//simple container for the array handle and functor
//so we can get them inside the array transfer class
template< class ArrayHandleType, class FunctorType>
struct ArrayPortalConstTransform
{
  DAX_CONT_EXPORT
  ArrayPortalConstTransform():
  Handle(),
  Functor()
  { }

  DAX_CONT_EXPORT
  ArrayPortalConstTransform(ArrayHandleType handle, FunctorType f):
  Handle(handle),
  Functor(f)
  { }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    return this->Handle.GetNumberOfValues();
  }

  ArrayHandleType Handle;
  FunctorType Functor;
};



template<class ValueType, class HandleType, class FunctorType>
struct ArrayContainerControlTagTransform { };

/// A convenience class that provides a typedef to the appropriate tag for
/// a counting array container.
template<typename ValueType, typename ArrayHandleType, typename FunctorType>
struct ArrayHandleTransformTraits
{
  typedef dax::cont::internal::ArrayContainerControlTagTransform<
                                                ValueType,
                                                ArrayHandleType,
                                                FunctorType > Tag;
  typedef dax::cont::internal::ArrayContainerControl<
          ValueType, Tag > ContainerType;

};


template<typename T, class ArrayHandleType, class FunctorType>
class ArrayContainerControl<
    T,
    ArrayContainerControlTagTransform<T,ArrayHandleType,FunctorType> >
{
public:

  typedef T ValueType;
  typedef ArrayPortalConstTransform<ArrayHandleType,FunctorType> PortalType;
  typedef PortalType PortalConstType;

public:
  DAX_CONT_EXPORT
  ArrayContainerControl() {
  }


  DAX_CONT_EXPORT
  PortalType GetPortal() {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }

  DAX_CONT_EXPORT
  PortalConstType GetPortalConst() const {
    throw dax::cont::ErrorControlBadValue(
          "Transform container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    throw dax::cont::ErrorControlBadValue(
          "Transform container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }

  DAX_CONT_EXPORT
  void Allocate(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlInternal(
      "The allocate method for the transform control array container should "
      "never have been called. The allocate is generally only called by "
      "the execution array manager, and the array transfer for the transform "
      "container should prevent the execution array manager from being "
      "directly used.");
  }

  DAX_CONT_EXPORT
  void Shrink(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }

  DAX_CONT_EXPORT
  void ReleaseResources() {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }
};


template<typename T,
         class ArrayHandleType,
         class FunctorType,
         class DeviceAdapterTag>
class ArrayTransfer<
    T,
    ArrayContainerControlTagTransform< T, ArrayHandleType, FunctorType >,
    DeviceAdapterTag>
{
private:
  typedef ArrayContainerControlTagTransform< T, ArrayHandleType, FunctorType>
      ArrayContainerControlTag;
  typedef dax::cont::internal::ArrayContainerControl<T,ArrayContainerControlTag>
      ContainerType;

public:
  typedef T ValueType;

  typedef typename ContainerType::PortalType PortalControl;
  typedef typename ContainerType::PortalConstType PortalConstControl;

  typedef ArrayPortalTransform< ValueType,
                  typename ArrayHandleType::PortalExecution,
                  FunctorType> PortalExecution;

  typedef ArrayPortalTransform< ValueType,
                  typename ArrayHandleType::PortalConstExecution,
                  FunctorType> PortalConstExecution;


  DAX_CONT_EXPORT
  ArrayTransfer() :
    PortalValid(false),
    NumberOfValues(0),
    Portal()
  {
  }

  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->NumberOfValues;
  }

  DAX_CONT_EXPORT void LoadDataForInput(PortalConstControl portal)
  {
    typename ArrayHandleType::PortalConstExecution tmpInput =
                           portal.Handle.PrepareForInput();
    this->NumberOfValues = portal.Handle.GetNumberOfValues();

    this->Portal = PortalConstExecution( tmpInput,
                                         this->NumberOfValues,
                                         portal.Functor );
    this->PortalValid = true;
  }

  DAX_CONT_EXPORT void LoadDataForInPlace(PortalControl daxNotUsed(portal))
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output or in place.");
  }

  DAX_CONT_EXPORT void AllocateArrayForOutput(
      ContainerType &daxNotUsed(controlArray),
      dax::Id daxNotUsed(numberOfValues))
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output.");
  }
  DAX_CONT_EXPORT void RetrieveOutputData(
      ContainerType &daxNotUsed(controlArray)) const
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output.");
  }

  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    DAX_ASSERT_CONT(this->PortalValid);
    std::copy(this->Portal.GetIteratorBegin(),
              this->Portal.GetIteratorEnd(),
              dest);
  }

  DAX_CONT_EXPORT void Shrink(dax::Id daxNotUsed(numberOfValues))
  {
    throw dax::cont::ErrorControlBadValue("Implicit arrays cannot be resized.");
  }

  DAX_CONT_EXPORT PortalExecution GetPortalExecution()
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays are read-only.  (Get the const portal.)");
  }
  DAX_CONT_EXPORT PortalConstExecution GetPortalConstExecution() const
  {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->Portal;
  }

  DAX_CONT_EXPORT void ReleaseResources() {  }

private:
  bool PortalValid;
  dax::Id NumberOfValues;
  PortalConstExecution Portal;

};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayHandleTransform_h
