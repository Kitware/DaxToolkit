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
#ifndef __dax_cont_ArrayContainerControlImplicit
#define __dax_cont_ArrayContainerControlImplicit

#include <dax/Types.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/Assert.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ArrayTransfer.h>

namespace dax {
namespace cont {

/// \brief An implementation for read-only implicit arrays.
///
/// It is sometimes the case that you want Dax to operate on an array of
/// implicit values. That is, rather than store the data in an actual array, it
/// is gerenated on the fly by a function. This is handled in Dax by creating
/// an ArrayHandle in Dax with an ArrayContainerControlTagImplicit type of
/// ArrayContainerControl. This tag itself is templated to specify an
/// ArrayPortal that generates the desired values. An ArrayHandle created with
/// this tag will raise an error on any operation that tries to modify it.
///
/// \todo The ArrayHandle currently copies the array in cases where the control
/// and environment do not share memory. This is wasteful and should be fixed.
///
template<class ArrayPortalType>
struct ArrayContainerControlTagImplicit {
  typedef ArrayPortalType PortalType;
};

namespace internal {

template<class ArrayPortalType>
class ArrayContainerControl<
    typename ArrayPortalType::ValueType,
    ArrayContainerControlTagImplicit<ArrayPortalType> >
{
public:
  typedef typename ArrayPortalType::ValueType ValueType;
  typedef ArrayPortalType PortalConstType;

  // This is meant to be invalid. Because implicit arrays are read only, you
  // should only be able to use the const version.
  struct PortalType {
    typedef void *ValueType;
    typedef void *IteratorType;
  };

  // All these methods do nothing but raise errors.
  PortalType GetPortal() {
    throw dax::cont::ErrorControlBadValue("Implicit arrays are read-only.");
  }
  PortalConstType GetPortalConst() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "Implicit container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  dax::Id GetNumberOfValues() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "Implicit container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  void Allocate(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Implicit arrays are read-only.");
  }
  void Shrink(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Implicit arrays are read-only.");
  }
  void ReleaseResources() {
    throw dax::cont::ErrorControlBadValue("Implicit arrays are read-only.");
  }
};

template<typename T, class ArrayPortalType, class DeviceAdapterTag>
class ArrayTransfer<
    T, ArrayContainerControlTagImplicit<ArrayPortalType>, DeviceAdapterTag>
{
private:
  typedef ArrayContainerControlTagImplicit<ArrayPortalType>
      ArrayContainerControlTag;
  typedef dax::cont::internal::ArrayContainerControl<T,ArrayContainerControlTag>
      ContainerType;

public:
  typedef T ValueType;

  typedef typename ContainerType::PortalType PortalControl;
  typedef typename ContainerType::PortalConstType PortalConstControl;
  typedef PortalControl PortalExecution;
  typedef PortalConstControl PortalConstExecution;

  ArrayTransfer() : PortalValid(false) {  }

  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->Portal.GetNumberOfValues();
  }

  DAX_CONT_EXPORT void LoadDataForInput(PortalControl portal) {
    this->Portal = portal;
    this->PortalValid = true;
  }
  DAX_CONT_EXPORT void LoadDataForInput(PortalConstControl portal) {
    this->Portal = portal;
    this->PortalValid = true;
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
  PortalConstExecution Portal;
  bool PortalValid;
};

} // namespace internal

}
} // namespace dax::cont

#endif //__dax_cont_ArrayContainerControlImplicit
