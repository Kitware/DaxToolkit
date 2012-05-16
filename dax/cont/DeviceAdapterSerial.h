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

#ifndef __dax_cont_DeviceAdapterSerial_h
#define __dax_cont_DeviceAdapterSerial_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

namespace dax { namespace cont {
// Forward declare the DeviceAdapter before the ArrayHandle uses it.
class DeviceAdapterSerial;
}}
#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cont::DeviceAdapterSerial

#include <dax/Functional.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/internal/ArrayManagerExecutionShareWithControl.h>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <numeric>

namespace dax {
namespace cont {
//forward declare the ArrayHandle before we use it.
template<typename T, template <typename> class ArrayContainerControl, class DeviceAdapter> class ArrayHandle;

/// A simple implementation of a DeviceAdapter that can be used for debuging.
/// The scheduling will simply run everything in a serial loop, which is easy
/// to track in a debugger.
///
struct DeviceAdapterSerial
{
  template <typename T, template <typename> class ArrayContainerControl>
  class ArrayManagerExecution
      : public dax::cont::internal::ArrayManagerExecutionShareWithControl
          <T, ArrayContainerControl>
  {
  public:
    typedef dax::cont::internal::ArrayManagerExecutionShareWithControl
        <T, ArrayContainerControl> Superclass;
    typedef typename Superclass::ValueType ValueType;
    typedef typename Superclass::IteratorType IteratorType;
    typedef typename Superclass::IteratorConstType IteratorConstType;
  };

  template <template <typename> class ArrayContainerControl>
  class ExecutionAdapter
  {
  public:
    template <typename T>
    struct FieldStructures
    {
      typedef typename DeviceAdapterSerial::ArrayManagerExecution<
          T,ArrayContainerControl>::IteratorType IteratorType;
      typedef typename DeviceAdapterSerial::ArrayManagerExecution<
          T,ArrayContainerControl>::IteratorConstType IteratorConstType;
    };

    class ErrorHandler
    {
    public:
      void RaiseError(const char *message) const
      {
        throw dax::cont::ErrorExecution(message);
      }
    };
  };

  template<typename T, template <typename> class Container>
  static void Copy(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& from,
      dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& to)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorType;
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstType;

    dax::Id numberOfValues = from.GetNumberOfValues();
    std::pair<IteratorConstType, IteratorConstType> fromIter
        = from.PrepareForInput();
    std::pair<IteratorType, IteratorType> toIter =
        to.PrepareForOutput(numberOfValues);

    std::copy(fromIter.first, fromIter.second, toIter.first);
    }

  template<typename T, template <typename> class Container>
  static T InclusiveScan(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial> &input,
      dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& output)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorType;
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstType;

    dax::Id numberOfValues = input.GetNumberOfValues();
    if (numberOfValues <= 0) { return 0; }

    std::pair<IteratorConstType, IteratorConstType> inputIter
        = input.PrepareForInput();
    std::pair<IteratorType, IteratorType> outputIter =
        output.PrepareForOutput(numberOfValues);

    std::partial_sum(inputIter.first, inputIter.second, outputIter.first);

    // Return the value at the last index in the array, which is the full sum.
    return *(outputIter.second - 1);
    }

  template<typename T, template <typename> class Container>
  static void LowerBounds(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& input,
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& values,
      dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterSerial>& output)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstT;
    typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorId;

    dax::Id numberOfValues = values.GetNumberOfValues();

    std::pair<IteratorConstT, IteratorConstT> inputIter
        = input.PrepareForInput();
    std::pair<IteratorConstT, IteratorConstT> valuesIter
        = values.PrepareForInput();
    std::pair<IteratorId, IteratorId> outputIter =
        output.PrepareForOutput(numberOfValues);

    // std::lower_bound only supports a single value to search for so iterate
    // over all values and search for each one.
    IteratorConstT value;
    IteratorId out;
    for (value = valuesIter.first, out = outputIter.first;
         value != valuesIter.second;
         value++, out++)
      {
      // std::lower_bound returns an iterator to the position where you can
      // insert, but we want the distance from the start.
      IteratorConstT resultPos
          = std::lower_bound(inputIter.first, inputIter.second, *value);
      *out = static_cast<dax::Id>(std::distance(inputIter.first, resultPos));
      }
    }

private: // Support methods/fields for Schedule
  // This runs in the execution environment.
  template<class FunctorType,
           class ParametersType,
           template <typename> class ArrayContainerControl>
  class ScheduleKernel
  {
  public:
    ScheduleKernel(
        const FunctorType &functor,
        const ParametersType &parameters)
      : Functor(functor),
        Parameters(parameters) {  }

    //needed for when calling from schedule on a range
    DAX_EXEC_EXPORT void operator()(dax::Id index)
    {
      this->Functor(this->Parameters,
                    index,
                    ExecutionAdapter<ArrayContainerControl>::ErrorHandler());
    }

  private:
    FunctorType Functor;
    ParametersType Parameters;
  };

public:

  template<class Functor,
           class Parameters,
           template <typename> class ArrayContainerControl>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances,
                       ExecutionAdapter<ArrayContainerControl>)
    {
    ScheduleKernel<Functor, Parameters, ArrayContainerControl>
        kernel(functor, parameters);

    std::for_each(
          ::boost::counting_iterator<dax::Id>(0),
          ::boost::counting_iterator<dax::Id>(numInstances),
          kernel);
    }


  template<typename T, template <typename> class Container>
  static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& values)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorType;

    std::pair<IteratorType, IteratorType> iterators
        = values.PrepareForInPlace();
    std::sort(iterators.first, iterators.second);
    }

  template<typename T, template <typename> class Container>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& input,
      dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterSerial>& output)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstT;
    typedef typename dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorId;

    std::pair<IteratorConstT, IteratorConstT> inputIter
        = input.PrepareForInput();

    dax::Id size = std::count_if(inputIter.first,
                                 inputIter.second,
                                 dax::not_default_constructor<dax::Id>());

    std::pair<IteratorId, IteratorId> outputIter
        = output.PrepareForOutput(size);

    IteratorConstT in = inputIter.first;
    IteratorId out = outputIter.first;
    dax::Id index = 0;
    for (; in != inputIter.second; in++, index++)
      {
      // Only write index that matches the default constructor of T
      if (dax::not_default_constructor<T>()(*in))
        {
        DAX_ASSERT_CONT(out != outputIter.second);
        *out = index;
        out++;
        }
      }
    DAX_ASSERT_CONT(out == outputIter.second);
    }

  template<typename T, typename U, template <typename> class Container>
  static void StreamCompact(
      const dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& input,
      const dax::cont::ArrayHandle<U,Container,DeviceAdapterSerial>& stencil,
      dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& output)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorT;
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstT;
    typedef typename dax::cont::ArrayHandle<U,Container,DeviceAdapterSerial>
        ::IteratorConstExecution IteratorConstU;

    std::pair<IteratorConstT, IteratorConstT> inputIter
        = input.PrepareForInput();
    std::pair<IteratorConstU, IteratorConstU> stencilIter
        = stencil.PrepareForInput();

    dax::Id size = std::count_if(stencilIter.first,
                                 stencilIter.second,
                                 dax::not_default_constructor<U>());

    std::pair<IteratorT, IteratorT> outputIter
        = output.PrepareForOutput(size);

    IteratorConstT in = inputIter.first;
    IteratorConstU flag = stencilIter.first;
    IteratorT out = outputIter.first;
    for (; in != inputIter.second; in++, flag++)
      {
      DAX_ASSERT_CONT(flag != stencilIter.second);
      // Only pass the input with a positive flag in the stencil.
      if (dax::not_default_constructor<U>()(*flag))
        {
        DAX_ASSERT_CONT(out != outputIter.second);
        *out = *in;
        out++;
        }
      }
    DAX_ASSERT_CONT(out == outputIter.second);
    }

  template<typename T, template <typename> class Container>
  static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>& values)
    {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterSerial>
        ::IteratorExecution IteratorT;

    std::pair<IteratorT, IteratorT> inputIter = values.PrepareForInPlace();

    IteratorT newEnd = std::unique(inputIter.first, inputIter.second);
    values.Shrink(std::distance(inputIter.first, newEnd));
    }

};

}
} // namespace dax::cont

#endif //__dax_cont_DeviceAdapterSerial_h
