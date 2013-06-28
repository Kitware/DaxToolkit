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
#include <dax/thrust/cont/internal/MakeThrustIterator.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>

#include <dax/Functional.h>

#include <dax/exec/internal/ArrayPortalFromIterators.h>
#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/Functor.h>

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

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <thrust/iterator/counting_iterator.h>

#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// This class can be subclassed to implement the DeviceAdapterAlgorithm for a
/// device that uses thrust as its implementation. The subclass should pass in
/// the correct device adapter tag as the template parameter.
///
template<class DeviceAdapterTag>
struct DeviceAdapterAlgorithmThrust
{
  // Because of some funny code conversions in nvcc, kernels for devices have to
  // be public.
  #ifndef DAX_CUDA
private:
  #endif
  template<class InputPortal, class OutputPortal>
  DAX_CONT_EXPORT static void CopyPortal(const InputPortal &input,
                                         const OutputPortal &output)
  {
    ::thrust::copy(IteratorBegin(input),
                   IteratorEnd(input),
                   IteratorBegin(output));
  }

  template<class InputPortal, class ValuesPortal, class OutputPortal>
  DAX_CONT_EXPORT static void LowerBoundsPortal(const InputPortal &input,
                                                const ValuesPortal &values,
                                                const OutputPortal &output)
  {
    ::thrust::lower_bound(IteratorBegin(input),
                          IteratorEnd(input),
                          IteratorBegin(values),
                          IteratorEnd(values),
                          IteratorBegin(output));
  }

  template<class InputPortal, class ValuesPortal, class OutputPortal,
           class Compare>
  DAX_CONT_EXPORT static void LowerBoundsPortal(const InputPortal &input,
                                                const ValuesPortal &values,
                                                const OutputPortal &output,
                                                Compare comp)
  {
    ::thrust::lower_bound(IteratorBegin(input),
                          IteratorEnd(input),
                          IteratorBegin(values),
                          IteratorEnd(values),
                          IteratorBegin(output),
                          comp);
  }

  template<class InputPortal, class OutputPortal>
  DAX_CONT_EXPORT static
  void LowerBoundsPortal(const InputPortal &input,
                         const OutputPortal &values_output)
  {
    ::thrust::lower_bound(IteratorBegin(input),
                          IteratorEnd(input),
                          IteratorBegin(values_output),
                          IteratorEnd(values_output),
                          IteratorBegin(values_output));
  }

  template<class InputPortal, class OutputPortal>
  DAX_CONT_EXPORT static
  typename InputPortal::ValueType ScanExclusivePortal(const InputPortal &input,
                                                      const OutputPortal &output)
  {
    // Use iterator to get value so that thrust device_ptr has chance to handle
    // data on device.
    typename InputPortal::ValueType inputEnd = *(IteratorEnd(input) - 1);

    ::thrust::exclusive_scan(IteratorBegin(input),
                             IteratorEnd(input),
                             IteratorBegin(output));

    //return the value at the last index in the array, as that is the sum
    return *(IteratorEnd(output) - 1) + inputEnd;
  }

  template<class InputPortal, class OutputPortal>
  DAX_CONT_EXPORT static
  typename InputPortal::ValueType ScanInclusivePortal(const InputPortal &input,
                                                      const OutputPortal &output)
  {
    ::thrust::inclusive_scan(IteratorBegin(input),
                             IteratorEnd(input),
                             IteratorBegin(output));

    //return the value at the last index in the array, as that is the sum
    return *(IteratorEnd(output) - 1);
  }

  template<class ValuesPortal>
  DAX_CONT_EXPORT static void SortPortal(const ValuesPortal &values)
  {
    ::thrust::sort(IteratorBegin(values),
                   IteratorEnd(values));
  }

  template<class ValuesPortal, class Compare>
  DAX_CONT_EXPORT static void SortPortal(const ValuesPortal &values,
                                         Compare comp)
  {
    ::thrust::sort(IteratorBegin(values),
                   IteratorEnd(values),
                   comp);
  }


  template<class KeysPortal, class ValuesPortal>
  DAX_CONT_EXPORT static void SortByKeyPortal(const KeysPortal &keys,
                                              const ValuesPortal &values)
  {
    ::thrust::sort_by_key(IteratorBegin(keys),
                          IteratorEnd(keys),
                          IteratorBegin(values));
  }

  template<class KeysPortal, class ValuesPortal, class Compare>
  DAX_CONT_EXPORT static void SortByKeyPortal(const KeysPortal &keys,
                                              const ValuesPortal &values,
                                              Compare comp)
  {
    ::thrust::sort_by_key(IteratorBegin(keys),
                          IteratorEnd(keys),
                          IteratorBegin(values),
                          comp);
  }

  template<class StencilPortal>
  DAX_CONT_EXPORT static dax::Id CountIfPortal(const StencilPortal &stencil)
  {
    typedef typename StencilPortal::ValueType ValueType;
    return ::thrust::count_if(IteratorBegin(stencil),
                              IteratorEnd(stencil),
                              dax::not_default_constructor<ValueType>());
  }

  template<class ValueIterator,
           class StencilPortal,
           class OutputPortal>
  DAX_CONT_EXPORT static void CopyIfPortal(ValueIterator valuesBegin,
                                           ValueIterator valuesEnd,
                                           const StencilPortal &stencil,
                                           const OutputPortal &output)
  {
    typedef typename StencilPortal::ValueType ValueType;
    ::thrust::copy_if(valuesBegin,
                      valuesEnd,
                      IteratorBegin(stencil),
                      IteratorBegin(output),
                      dax::not_default_constructor<ValueType>());
  }

  template<class ValueIterator,
           class StencilArrayHandle,
           class OutputArrayHandle>
  DAX_CONT_EXPORT static void RemoveIf(ValueIterator valuesBegin,
                                       ValueIterator valuesEnd,
                                       const StencilArrayHandle& stencil,
                                       OutputArrayHandle& output)
  {
    dax::Id numLeft = CountIfPortal(stencil.PrepareForInput());

    CopyIfPortal(valuesBegin,
                 valuesEnd,
                 stencil.PrepareForInput(),
                 output.PrepareForOutput(numLeft));
  }

  template<class InputPortal,
           class StencilArrayHandle,
           class OutputArrayHandle>
  DAX_CONT_EXPORT static
  void StreamCompactPortal(const InputPortal& inputPortal,
                           const StencilArrayHandle &stencil,
                           OutputArrayHandle& output)
  {
    RemoveIf(IteratorBegin(inputPortal),
             IteratorEnd(inputPortal),
             stencil,
             output);
  }

  template<class ValuesPortal>
  DAX_CONT_EXPORT static
  dax::Id UniquePortal(const ValuesPortal values)
  {
    typedef typename detail::IteratorTraits<ValuesPortal>::IteratorType
                                                            IteratorType;
    IteratorType begin = IteratorBegin(values);
    IteratorType newLast = ::thrust::unique(begin, IteratorEnd(values));
    return ::thrust::distance(begin, newLast);
  }

  template<class ValuesPortal, class Compare>
  DAX_CONT_EXPORT static
  dax::Id UniquePortal(const ValuesPortal values, Compare comp)
  {
    typedef typename detail::IteratorTraits<ValuesPortal>::IteratorType
                                                            IteratorType;
    IteratorType begin = IteratorBegin(values);
    IteratorType newLast = ::thrust::unique(begin, IteratorEnd(values), comp);
    return ::thrust::distance(begin, newLast);
  }

  template<class InputPortal, class ValuesPortal, class OutputPortal>
  DAX_CONT_EXPORT static
  void UpperBoundsPortal(const InputPortal &input,
                         const ValuesPortal &values,
                         const OutputPortal &output)
  {
    ::thrust::upper_bound(IteratorBegin(input),
                          IteratorEnd(input),
                          IteratorBegin(values),
                          IteratorEnd(values),
                          IteratorBegin(output));
  }

  template<class InputPortal, class OutputPortal>
  DAX_CONT_EXPORT static
  void UpperBoundsPortal(const InputPortal &input,
                         const OutputPortal &values_output)
  {
    ::thrust::upper_bound(IteratorBegin(input),
                          IteratorEnd(input),
                          IteratorBegin(values_output),
                          IteratorEnd(values_output),
                          IteratorBegin(values_output));
  }

//-----------------------------------------------------------------------------

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static void Copy(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag> &output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();
    CopyPortal(input.PrepareForInput(),
               output.PrepareForOutput(numberOfValues));
  }

  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>& values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = values.GetNumberOfValues();
    LowerBoundsPortal(input.PrepareForInput(),
                      values.PrepareForInput(),
                      output.PrepareForOutput(numberOfValues));
  }

  template<typename T, class CIn, class CVal, class COut, class Compare>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>& values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output,
      Compare comp)
  {
    dax::Id numberOfValues = values.GetNumberOfValues();
    LowerBoundsPortal(input.PrepareForInput(),
                      values.PrepareForInput(),
                      output.PrepareForOutput(numberOfValues),
                      comp);
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    LowerBoundsPortal(input.PrepareForInput(),
                      values_output.PrepareForInPlace());
  }

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();
    if (numberOfValues <= 0)
      {
      output.PrepareForOutput(0);
      return 0;
      }

    return ScanExclusivePortal(input.PrepareForInput(),
                               output.PrepareForOutput(numberOfValues));
  }
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanInclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();
    if (numberOfValues <= 0)
      {
      output.PrepareForOutput(0);
      return 0;
      }

    return ScanInclusivePortal(input.PrepareForInput(),
                               output.PrepareForOutput(numberOfValues));
  }

// Because of some funny code conversions in nvcc, kernels for devices have to
// be public.
#ifndef DAX_CUDA
private:
#endif
  template<class FunctorType>
  class ScheduleKernel
  {
  public:
    DAX_CONT_EXPORT ScheduleKernel(const FunctorType &functor)
      : Functor(functor)
    {  }

    DAX_EXEC_EXPORT void operator()(dax::Id index) const {
      this->Functor(index);
    }
  private:
    FunctorType Functor;
  };

public:
  template<class Functor>
  DAX_CONT_EXPORT static void Schedule(Functor functor, dax::Id numInstances)
  {
    const dax::Id ERROR_ARRAY_SIZE = 1024;
    ::thrust::device_vector<char> errorArray(ERROR_ARRAY_SIZE);
    errorArray[0] = '\0';
    dax::exec::internal::ErrorMessageBuffer errorMessage(
          ::thrust::raw_pointer_cast(&(*errorArray.begin())),
          errorArray.size());

    functor.SetErrorMessageBuffer(errorMessage);

    ScheduleKernel<Functor> kernel(functor);

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

  template<class FunctorType>
  DAX_CONT_EXPORT
  static void Schedule(FunctorType functor, const dax::Id3& rangeMax)
  {
    //default behavior for the general algorithm is to defer to the default
    //schedule implementation. if you want to customize schedule for certain
    //grid types, you need to specialize this method
    typedef DeviceAdapterAlgorithmThrust<DeviceAdapterTag> DAAT;
    DAAT::Schedule(functor, rangeMax[0]*rangeMax[1]*rangeMax[2]);
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>& values)
  {
    SortPortal(values.PrepareForInPlace());
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>& values,
      Compare comp)
  {
    SortPortal(values.PrepareForInPlace(),comp);
  }

  template<typename T, typename U,
           class ContainerT, class ContainerU>
  DAX_CONT_EXPORT static void SortByKey(
      dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag>& keys,
      dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag>& values)
  {
    SortByKeyPortal(keys.PrepareForInPlace(),
                    values.PrepareForInPlace());
  }

  template<typename T, typename U,
           class ContainerT, class ContainerU,
           class Compare>
  DAX_CONT_EXPORT static void SortByKey(
      dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag>& keys,
      dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag>& values,
      Compare comp)
  {
    SortByKeyPortal(keys.PrepareForInPlace(),
                    values.PrepareForInPlace(),
                    comp);
  }


  template<typename T, class CStencil, class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag>& stencil,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id stencilSize = stencil.GetNumberOfValues();

    RemoveIf(::thrust::make_counting_iterator<dax::Id>(0),
             ::thrust::make_counting_iterator<dax::Id>(stencilSize),
             stencil,
             output);
  }

  template<typename T,
           typename U,
           class CIn,
           class CStencil,
           class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<U,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag>& stencil,
      dax::cont::ArrayHandle<U,COut,DeviceAdapterTag>& output)
  {
    StreamCompactPortal(input.PrepareForInput(), stencil, output);
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values)
  {
    dax::Id newSize = UniquePortal(values.PrepareForInPlace());

    values.Shrink(newSize);
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values,
      Compare comp)
  {
    dax::Id newSize = UniquePortal(values.PrepareForInPlace(),comp);

    values.Shrink(newSize);
  }

  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>& values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = values.GetNumberOfValues();
    UpperBoundsPortal(input.PrepareForInput(),
                      values.PrepareForInput(),
                      output.PrepareForOutput(numberOfValues));
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    UpperBoundsPortal(input.PrepareForInput(),
                      values_output.PrepareForInPlace());
  }
};

}
}
}
} // namespace dax::thrust::cont::internal

#endif //__dax_thrust_cont_internal_DeviceAdapterThrust_h
