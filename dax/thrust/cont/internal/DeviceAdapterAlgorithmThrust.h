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
#ifndef __dax_thrust_cont_internal_DeviceAdapterThrust_h
#define __dax_thrust_cont_internal_DeviceAdapterThrust_h

#include <dax/thrust/cont/internal/CheckThrustBackend.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>

#include <dax/Functional.h>

#include <dax/exec/internal/ArrayPortalFromIterators.h>
#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/Functor.h>

#include <dax/thrust/cont/internal/DeviceAdapterTagThrust.h>

// Disable GCC warnings we check Dax for but Thrust does not.
#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

// These have to be in the dax::cont::internal namespace to match those
// defined elsewhere.
namespace dax {
namespace cont {
namespace internal {

namespace detail {

template<class PortalType>
class ThrustLookupFunctor
    : public ::thrust::unary_function<dax::Id, typename PortalType::ValueType>
{
public:
  DAX_CONT_EXPORT ThrustLookupFunctor(PortalType portal)
    : Portal(portal) {  }

  DAX_EXEC_EXPORT
  typename PortalType::ValueType
  operator()(dax::Id index)
  {
    return this->Portal.Get(index);
  }

private:
  PortalType Portal;
};

// Tags to specify what type of thrust iterator to use.
struct ThrustIteratorTransformTag {  };
struct ThrustIteratorDevicePtrTag {  };

// Traits to help classify what thrust iterators will be used.
template<class IteratorType>
struct ThrustIteratorTag {
  typedef ThrustIteratorTransformTag Type;
};
template<typename T>
struct ThrustIteratorTag<T *> {
  typedef ThrustIteratorDevicePtrTag Type;
};
template<typename T>
struct ThrustIteratorTag<const T*> {
  typedef ThrustIteratorDevicePtrTag Type;
};

template<typename T> struct ThrustStripPointer;
template<typename T> struct ThrustStripPointer<T *> {
  typedef T Type;
};
template<typename T> struct ThrustStripPointer<const T *> {
  typedef const T Type;
};

template<class PortalType, class Tag> struct ThrustIteratorType;
template<class PortalType>
struct ThrustIteratorType<PortalType, ThrustIteratorTransformTag> {
  typedef ::thrust::transform_iterator<
      ThrustLookupFunctor<PortalType>,
      ::thrust::counting_iterator<dax::Id> > Type;
};
template<class PortalType>
struct ThrustIteratorType<PortalType, ThrustIteratorDevicePtrTag> {
  typedef ::thrust::device_ptr<
      typename ThrustStripPointer<typename PortalType::IteratorType>::Type>
      Type;
};

template<class PortalType>
struct ThrustIteratorTraits
{
  typedef typename PortalType::IteratorType BaseIteratorType;
  typedef typename ThrustIteratorTag<BaseIteratorType>::Type Tag;
  typedef typename ThrustIteratorType<PortalType, Tag>::Type IteratorType;
};

template<typename T>
DAX_CONT_EXPORT
::thrust::device_ptr<T>
ThrustMakeDevicePtr(T *iter)
{
  return ::thrust::device_ptr<T>(iter);
}
template<typename T>
DAX_CONT_EXPORT
::thrust::device_ptr<const T>
ThrustMakeDevicePtr(const T *iter)
{
  return ::thrust::device_ptr<const T>(iter);
}

template<class PortalType>
DAX_CONT_EXPORT
typename ThrustIteratorTraits<PortalType>::IteratorType
ThrustMakeIteratorBegin(PortalType portal, ThrustIteratorTransformTag)
{
  return ::thrust::make_transform_iterator(
        ::thrust::make_counting_iterator(dax::Id(0)),
        ThrustLookupFunctor<PortalType>(portal));
}

template<class PortalType>
DAX_CONT_EXPORT
typename ThrustIteratorTraits<PortalType>::IteratorType
ThrustMakeIteratorBegin(PortalType portal, ThrustIteratorDevicePtrTag)
{
  return ThrustMakeDevicePtr(portal.GetIteratorBegin());
}

template<class PortalType>
DAX_CONT_EXPORT
typename ThrustIteratorTraits<PortalType>::IteratorType
ThrustIteratorBegin(PortalType portal)
{
  typedef typename ThrustIteratorTraits<PortalType>::Tag ThrustIteratorTag;
  return ThrustMakeIteratorBegin(portal, ThrustIteratorTag());
}
template<class PortalType>
DAX_CONT_EXPORT
typename ThrustIteratorTraits<PortalType>::IteratorType
ThrustIteratorEnd(PortalType portal)
{
  return ThrustIteratorBegin(portal) + portal.GetNumberOfValues();
}

} // namespace detail

template<typename T, class CIn, class COut, class Adapter>
DAX_CONT_EXPORT void Copy(
    const dax::cont::ArrayHandle<T,CIn,Adapter> &input,
    dax::cont::ArrayHandle<T,COut,Adapter> &output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>
      InputManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<T,COut,Adapter>
      OutputManagerType;

  dax::Id numberOfValues = input.GetNumberOfValues();
  typename InputManagerType::PortalConstType inputPortal =
      input.PrepareForInput();
  typename OutputManagerType::PortalType outputPortal =
      output.PrepareForOutput(numberOfValues);

  ::thrust::copy(detail::ThrustIteratorBegin(inputPortal),
                 detail::ThrustIteratorEnd(inputPortal),
                 detail::ThrustIteratorBegin(outputPortal));
}

template<typename T, class CIn, class COut, class Adapter>
DAX_CONT_EXPORT T InclusiveScan(
    const dax::cont::ArrayHandle<T,CIn,Adapter> &input,
    dax::cont::ArrayHandle<T,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>
      InputManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<T,COut,Adapter>
      OutputManagerType;

  dax::Id numberOfValues = input.GetNumberOfValues();

  typename InputManagerType::PortalConstType inputPortal =
      input.PrepareForInput();
  typename OutputManagerType::PortalType outputPortal =
      output.PrepareForOutput(numberOfValues);

  if (numberOfValues <= 0) { return 0; }

  ::thrust::inclusive_scan(detail::ThrustIteratorBegin(inputPortal),
                           detail::ThrustIteratorEnd(inputPortal),
                           detail::ThrustIteratorBegin(outputPortal));

  //return the value at the last index in the array, as that is the size
  return *(detail::ThrustIteratorEnd(outputPortal) - 1);
}

template<typename T, class CIn, class CVal, class COut, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<T,CIn,Adapter>& input,
    const dax::cont::ArrayHandle<T,CVal,Adapter>& values,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,CIn,Adapter>
      InputManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<T,CVal,Adapter>
      ValueManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<dax::Id,COut,Adapter>
      OutputManagerType;

  dax::Id numberOfValues = values.GetNumberOfValues();

  typename InputManagerType::PortalConstType inputPortal =
      input.PrepareForInput();
  typename ValueManagerType::PortalConstType valuesPortal =
      values.PrepareForInput();
  typename OutputManagerType::PortalType outputPortal =
      output.PrepareForOutput(numberOfValues);

  ::thrust::lower_bound(detail::ThrustIteratorBegin(inputPortal),
                        detail::ThrustIteratorEnd(inputPortal),
                        detail::ThrustIteratorBegin(valuesPortal),
                        detail::ThrustIteratorEnd(valuesPortal),
                        detail::ThrustIteratorBegin(outputPortal));
}

template<class CIn, class COut, class Adapter>
DAX_CONT_EXPORT void LowerBounds(
    const dax::cont::ArrayHandle<dax::Id,CIn,Adapter> &input,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter> &values_output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<dax::Id,CIn,Adapter>
      InputManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<dax::Id,COut,Adapter>
      OutputManagerType;

  typename InputManagerType::PortalConstType inputPortal =
      input.PrepareForInput();
  typename OutputManagerType::PortalType outputPortal =
      values_output.PrepareForInPlace();

  ::thrust::lower_bound(detail::ThrustIteratorBegin(inputPortal),
                        detail::ThrustIteratorEnd(inputPortal),
                        detail::ThrustIteratorBegin(outputPortal),
                        detail::ThrustIteratorEnd(outputPortal),
                        detail::ThrustIteratorBegin(outputPortal));
}

namespace detail {

template<class FunctorType>
class ScheduleKernelThrust
{
public:
  DAX_CONT_EXPORT ScheduleKernelThrust(const FunctorType &functor)
    : Functor(functor)
  {  }

  DAX_EXEC_EXPORT void operator()(dax::Id index) const {
    this->Functor(index);
  }

private:
  FunctorType Functor;
};

} // namespace detail

template<class Functor>
DAX_CONT_EXPORT void LegacySchedule(
    Functor functor,
    dax::Id numInstances,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  ::thrust::device_vector<char> errorArray(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer errorMessage(
        ::thrust::raw_pointer_cast(&(*errorArray.begin())),
        errorArray.size());

  functor.SetErrorMessageBuffer(errorMessage);

  detail::ScheduleKernelThrust<Functor> kernel(functor);

  ::thrust::for_each(::thrust::make_counting_iterator<dax::Id>(0),
                     ::thrust::make_counting_iterator<dax::Id>(numInstances),
                     kernel);

  if (errorArray[0] != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    ::thrust::copy(errorArray.begin(), errorArray.end(), errorString);

    throw dax::cont::ErrorExecution(errorString);
    }
}

template<class ControlInvocSig, class Functor,  class Bindings>
DAX_CONT_EXPORT void Schedule(
    Functor functor,
    Bindings& bindings,
    dax::Id numInstances,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  //setup error message
  const dax::Id ERROR_ARRAY_SIZE = 1024;
  ::thrust::device_vector<char> errorArray(ERROR_ARRAY_SIZE);
  errorArray[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer errorMessage(
        ::thrust::raw_pointer_cast(&(*errorArray.begin())),
        errorArray.size());

  functor.SetErrorMessageBuffer(errorMessage);

  //setup functor
  dax::exec::internal::Functor<ControlInvocSig> kernel(functor, bindings);
  ::thrust::for_each(::thrust::make_counting_iterator<dax::Id>(0),
                     ::thrust::make_counting_iterator<dax::Id>(numInstances),
                     kernel);

  if (errorArray[0] != '\0')
    {
    char errorString[ERROR_ARRAY_SIZE];
    ::thrust::copy(errorArray.begin(), errorArray.end(), errorString);

    throw dax::cont::ErrorExecution(errorString);
    }
}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Sort(
    dax::cont::ArrayHandle<T,Container,Adapter>& values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      ManagerType;

  typename ManagerType::PortalType portal = values.PrepareForInPlace();

  ::thrust::sort(detail::ThrustIteratorBegin(portal),
                 detail::ThrustIteratorEnd(portal));
}


namespace detail {

template<class ValueIterator,
         typename T,
         typename U,
         class CStencil,
         class COut,
         class Adapter>
DAX_CONT_EXPORT void RemoveIf(
    ValueIterator valuesBegin,
    ValueIterator valuesEnd,
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<U,COut,Adapter>& output)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,CStencil,Adapter>
      StencilManagerType;
  typedef dax::cont::internal::ArrayManagerExecution<U,COut,Adapter>
      OutputManagerType;

  typename StencilManagerType::PortalConstType stencilPortal =
      stencil.PrepareForInput();

  dax::Id numLeft = ::thrust::count_if(
        detail::ThrustIteratorBegin(stencilPortal),
        detail::ThrustIteratorEnd(stencilPortal),
        dax::not_default_constructor<T>());

  typename OutputManagerType::PortalType outputPortal =
      output.PrepareForOutput(numLeft);

  ::thrust::copy_if(valuesBegin,
                    valuesEnd,
                    detail::ThrustIteratorBegin(stencilPortal),
                    detail::ThrustIteratorBegin(outputPortal),
                    dax::not_default_constructor<T>());
}

} // namespace detail

template<typename T, class CStencil, class COut, class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<dax::Id,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  dax::Id stencilSize = stencil.GetNumberOfValues();

  detail::RemoveIf(::thrust::make_counting_iterator<dax::Id>(0),
                   ::thrust::make_counting_iterator<dax::Id>(stencilSize),
                   stencil,
                   output);
}

template<typename T,
         typename U,
         class CIn,
         class CStencil,
         class COut,
         class Adapter>
DAX_CONT_EXPORT void StreamCompact(
    const dax::cont::ArrayHandle<U,CIn,Adapter>& input,
    const dax::cont::ArrayHandle<T,CStencil,Adapter>& stencil,
    dax::cont::ArrayHandle<U,COut,Adapter>& output,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<U,CIn,Adapter>
      InputManagerType;

  typename InputManagerType::PortalConstType inputPortal =
      input.PrepareForInput();

  detail::RemoveIf(detail::ThrustIteratorBegin(inputPortal),
                   detail::ThrustIteratorEnd(inputPortal),
                   stencil,
                   output);
}

namespace detail {

// A simple wrapper around unique that returns the size of the array.
// This would not be necessary if we had an auto keyword.
template<class IteratorType>
DAX_CONT_EXPORT
dax::Id ThrustUnique(IteratorType first, IteratorType last)
{
  IteratorType newLast = ::thrust::unique(first, last);
  return ::thrust::distance(first, newLast);
}

}

template<typename T, class Container, class Adapter>
DAX_CONT_EXPORT void Unique(
    dax::cont::ArrayHandle<T,Container,Adapter> &values,
    dax::thrust::cont::internal::DeviceAdapterTagThrust)
{
  typedef dax::cont::internal::ArrayManagerExecution<T,Container,Adapter>
      ManagerType;

  typename ManagerType::PortalType valuePortal = values.PrepareForInPlace();

  dax::Id newSize =
      detail::ThrustUnique(detail::ThrustIteratorBegin(valuePortal),
                           detail::ThrustIteratorEnd(valuePortal));

  values.Shrink(newSize);
}

}
}
} // namespace dax::cont::internal

#endif //__dax_thrust_cont_internal_DeviceAdapterThrust_h
