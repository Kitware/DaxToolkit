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
#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

#include <dax/Functional.h>

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

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

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
private:
  template<typename T, class Container>
  struct ThrustIter {
    typedef std::pair<
        typename dax::cont::internal::
            ArrayManagerExecution<T,Container,DeviceAdapterTag>::
            ThrustIteratorType,
        typename dax::cont::internal::
            ArrayManagerExecution<T,Container,DeviceAdapterTag>::
            ThrustIteratorType>
        Type;
  };

  template<typename T, class Container>
  struct ThrustIterConst {
    typedef std::pair<
        typename dax::cont::internal::
            ArrayManagerExecution<T,Container,DeviceAdapterTag>::
            ThrustIteratorConstType,
        typename dax::cont::internal::
            ArrayManagerExecution<T,Container,DeviceAdapterTag>::
            ThrustIteratorConstType>
        Type;
  };

  template<typename T, class Container>
  struct ChooseManager {
    typedef dax::cont::internal::ArrayManagerExecution<
          T,Container,DeviceAdapterTag>
        Type;
  };

  template<typename T, class Container>
  DAX_CONT_EXPORT static
  typename ThrustIterConst<T,Container>::Type
  PrepareForInput(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &array)
  {
    typedef typename ChooseManager<T,Container>::Type Manager;
    typedef typename Manager::PortalConstType PortalConstType;

    PortalConstType portal = array.PrepareForInput();
    return std::make_pair(Manager::ThrustIteratorBegin(portal),
                          Manager::ThrustIteratorEnd(portal));
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static
  typename ThrustIter<T,Container>::Type
  PrepareForOutput(dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &array,
                   dax::Id numberOfValues)
  {
    typedef typename ChooseManager<T,Container>::Type Manager;
    typedef typename Manager::PortalType PortalType;

    PortalType portal = array.PrepareForOutput(numberOfValues);
    return std::make_pair(Manager::ThrustIteratorBegin(portal),
                          Manager::ThrustIteratorEnd(portal));
  }

  template<typename T, class Container, class Adapter>
  DAX_CONT_EXPORT static
  typename ThrustIter<T,Container>::Type
  PrepareForInPlace(dax::cont::ArrayHandle<T,Container,Adapter> &array)
  {
    typedef typename ChooseManager<T,Container>::Type Manager;
    typedef typename Manager::PortalType PortalType;

    PortalType portal = array.PrepareForInPlace();
    return std::make_pair(Manager::ThrustIteratorBegin(portal),
                          Manager::ThrustIteratorEnd(portal));
  }

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static void Copy(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag> &output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();
    typename ThrustIterConst<T,CIn>::Type inputIter = PrepareForInput(input);
    typename ThrustIter<T,COut>::Type outputIter = PrepareForOutput(
                                                     output, numberOfValues);

    ::thrust::copy(inputIter.first, inputIter.second, outputIter.first);
  }

  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>& values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = values.GetNumberOfValues();

    typename ThrustIterConst<T,CIn>::Type inputIter = PrepareForInput(input);
    typename ThrustIterConst<T,CVal>::Type valuesIter = PrepareForInput(values);
    typename ThrustIter<dax::Id,COut>::Type outputIter =
        PrepareForOutput(output, numberOfValues);

    ::thrust::lower_bound(inputIter.first,
                          inputIter.second,
                          valuesIter.first,
                          valuesIter.second,
                          outputIter.first);
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    typename ThrustIterConst<dax::Id,CIn>::Type inputIter =
        PrepareForInput(input);
    typename ThrustIter<dax::Id,COut>::Type outputIter =
        PrepareForInPlace(values_output);

    ::thrust::lower_bound(inputIter.first,
                          inputIter.second,
                          outputIter.first,
                          outputIter.second,
                          outputIter.first);
  }

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanInclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();

    typename ThrustIterConst<T,CIn>::Type inputIter = PrepareForInput(input);
    typename ThrustIter<T,COut>::Type outputIter = PrepareForOutput(
                                                     output, numberOfValues);

    if (numberOfValues <= 0) { return 0; }

    typename ThrustIter<T,COut>::Type::first_type result =
        ::thrust::inclusive_scan(inputIter.first,
                                 inputIter.second,
                                 outputIter.first);

    //return the value at the last index in the array, as that is the size
    return *(result - 1);
  }

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = input.GetNumberOfValues();

    typename ThrustIterConst<T,CIn>::Type inputIter = PrepareForInput(input);
    typename ThrustIter<T,COut>::Type outputIter = PrepareForOutput(
                                                     output, numberOfValues);

    if (numberOfValues <= 0) { return 0; }

    typename ThrustIter<T,COut>::Type::first_type result =
        ::thrust::exclusive_scan(inputIter.first,
                                 inputIter.second,
                                 outputIter.first);
    return *(result - 1);
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


  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>& values)
  {
    typename ThrustIter<T,Container>::Type iterators =
        PrepareForInPlace(values);
    ::thrust::sort(iterators.first, iterators.second);
  }


private:
  template<class ValueIterator,
           typename T,
           typename U,
           class CStencil,
           class COut>
  DAX_CONT_EXPORT static void RemoveIf(
      ValueIterator valuesBegin,
      ValueIterator valuesEnd,
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag>& stencil,
      dax::cont::ArrayHandle<U,COut,DeviceAdapterTag>& output)
  {
    typename ThrustIterConst<T,CStencil>::Type stencilIter =
        PrepareForInput(stencil);

    dax::Id numLeft = ::thrust::count_if(stencilIter.first,
                                         stencilIter.second,
                                         dax::not_default_constructor<T>());

    typename ThrustIter<U,COut>::Type outputIter =
        PrepareForOutput(output, numLeft);

    ::thrust::copy_if(valuesBegin,
                      valuesEnd,
                      stencilIter.first,
                      outputIter.first,
                      dax::not_default_constructor<T>());
  }

public:
  template<typename T, class CStencil, class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag>& stencil,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id stencilSize = stencil.GetNumberOfValues();

    DeviceAdapterAlgorithmThrust::RemoveIf(
          ::thrust::make_counting_iterator<dax::Id>(0),
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
    typename ThrustIterConst<U,CIn>::Type inputIter = PrepareForInput(input);

    DeviceAdapterAlgorithmThrust::RemoveIf(inputIter.first,
                                           inputIter.second,
                                           stencil,
                                           output);
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values)
  {
    typename ThrustIter<T,Container>::Type valueIter =
        PrepareForInPlace(values);

    typename ThrustIter<T,Container>::Type::first_type newEnd =
        ::thrust::unique(valueIter.first, valueIter.second);

    values.Shrink(::thrust::distance(valueIter.first, newEnd));
  }

  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>& values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>& output)
  {
    dax::Id numberOfValues = values.GetNumberOfValues();

    typename ThrustIterConst<T,CIn>::Type inputIter = PrepareForInput(input);
    typename ThrustIterConst<T,CVal>::Type valuesIter = PrepareForInput(values);
    typename ThrustIter<dax::Id,COut>::Type outputIter =
        PrepareForOutput(output, numberOfValues);

    ::thrust::upper_bound(inputIter.first,
                          inputIter.second,
                          valuesIter.first,
                          valuesIter.second,
                          outputIter.first);
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    typename ThrustIterConst<dax::Id,CIn>::Type inputIter =
        PrepareForInput(input);
    typename ThrustIter<dax::Id,COut>::Type outputIter =
        PrepareForInPlace(values_output);

    ::thrust::upper_bound(inputIter.first,
                          inputIter.second,
                          outputIter.first,
                          outputIter.second,
                          outputIter.first);
  }

};

}
}
}
} // namespace dax::thrust::cont::internal

#endif //__dax_thrust_cont_internal_DeviceAdapterThrust_h
