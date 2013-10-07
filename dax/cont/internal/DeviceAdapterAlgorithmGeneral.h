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
#ifndef __dax_cont_internal_DeviceAdapterAlgorithmGeneral_h
#define __dax_cont_internal_DeviceAdapterAlgorithmGeneral_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleConstant.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/internal/ArrayContainerControlCounting.h>
#include <dax/cont/internal/ArrayHandleZip.h>

#include <dax/Functional.h>

#include <dax/exec/Assert.h>
#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/WorkletBase.h>

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>

#include <algorithm>

namespace dax {
namespace cont {
namespace internal {

/// \brief
///
/// This struct provides algorithms that implement "general" device adapter
/// algorithms. If a device adapter provides implementations for Schedule,
/// and Synchronize, the rest of the algorithms can be implemented by calling
/// these functions.
///
/// It should be noted that we recommend that you also implement Sort,
/// ScanInclusive, and ScanExclusive for improved performance.
///
/// An easy way to implement the DeviceAdapterAlgorithm specialization is to
/// subclass this and override the implementation of methods as necessary.
/// As an example, the code would look something like this.
///
/// \code{.cpp}
/// template<>
/// struct DeviceAdapterAlgorithm<DeviceAdapterTagFoo>
///    : DeviceAdapterAlgorithmGeneral<DeviceAdapterAlgorithm<DeviceAdapterTagFoo>,
///                                    DeviceAdapterTagFoo>
/// {
///   template<class Functor>
///   DAX_CONT_EXPORT static void Schedule(Functor functor,
///                                        dax::Id numInstances)
///   {
///     ...
///   }
///
///   template<class Functor>
///   DAX_CONT_EXPORT static void Schedule(Functor functor,
///                                        dax::Id3 maxRange)
///   {
///     ...
///   }
///
///   DAX_CONT_EXPORT static void Synchronize()
///   {
///     ...
///   }
/// };
/// \endcode
///
/// You might note that DeviceAdapterAlgorithmGeneral has two template
/// parameters that are redundant. Although the first parameter, the class for
/// the actual DeviceAdapterAlgorithm class containing Schedule, and
/// Synchronize is the same as DeviceAdapterAlgorithm<DeviceAdapterTag>, it is
/// made a separate template parameter to avoid a recursive dependence between
/// DeviceAdapterAlgorithmGeneral.h and DeviceAdapterAlgorithm.h
///
template<class DerivedAlgorithm, class DeviceAdapterTag>
struct DeviceAdapterAlgorithmGeneral
{
  //--------------------------------------------------------------------------
  // Get Execution Value
  // This method is used internally to get a single element from the execution
  // array. Might want to expose this and/or allow actual device adapter
  // implementations to provide one.
private:
  template<typename T, class CIn>
  DAX_CONT_EXPORT
  static T GetExecutionValue(
      const dax::cont::ArrayHandle<T, CIn, DeviceAdapterTag> &input,
      dax::Id index)
  {
    typedef dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> InputArrayType;
    typedef dax::cont::ArrayHandle<
        T,dax::cont::ArrayContainerControlTagBasic,DeviceAdapterTag>
        OutputArrayType;

    OutputArrayType output;

    CopyKernel<
        typename InputArrayType::PortalConstExecution,
        typename OutputArrayType::PortalExecution>
        kernel(input.PrepareForInput(),
               output.PrepareForOutput(1),
               index);

    DerivedAlgorithm::Schedule(kernel, 1);

    return output.GetPortalConstControl().Get(0);
  }

  //--------------------------------------------------------------------------
  // Copy
private:
  template<class InputPortalType, class OutputPortalType>
  struct CopyKernel {
    InputPortalType InputPortal;
    OutputPortalType OutputPortal;
    dax::Id InputOffset;
    dax::Id OutputOffset;

    DAX_CONT_EXPORT
    CopyKernel(InputPortalType inputPortal,
               OutputPortalType outputPortal,
               dax::Id inputOffset = 0,
               dax::Id outputOffset = 0)
      : InputPortal(inputPortal),
        OutputPortal(outputPortal),
        InputOffset(inputOffset),
        OutputOffset(outputOffset)
    {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      this->OutputPortal.Set(
            index + this->OutputOffset,
            this->InputPortal.Get(index + this->InputOffset));
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static void Copy(
      const dax::cont::ArrayHandle<T, CIn, DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T, COut, DeviceAdapterTag> &output)
  {
    dax::Id arraySize = input.GetNumberOfValues();

    CopyKernel<
        typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>::PortalExecution>
        kernel(input.PrepareForInput(),
               output.PrepareForOutput(arraySize));

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  //--------------------------------------------------------------------------
  // Lower Bounds
private:
  template<class InputPortalType,class ValuesPortalType,class OutputPortalType>
  struct LowerBoundsKernel {
    InputPortalType InputPortal;
    ValuesPortalType ValuesPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    LowerBoundsKernel(InputPortalType inputPortal,
                      ValuesPortalType valuesPortal,
                      OutputPortalType outputPortal)
      : InputPortal(inputPortal),
        ValuesPortal(valuesPortal),
        OutputPortal(outputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      // This method assumes that (1) InputPortalType can return working
      // iterators in the execution environment and that (2) methods not
      // specified with DAX_EXEC_EXPORT (such as the STL algorithms) can be
      // called from the execution environment. Neither one of these is
      // necessarily true, but it is true for the current uses of this general
      // function and I don't want to compete with STL if I don't have to.

      typename InputPortalType::IteratorType resultPos =
          std::lower_bound(this->InputPortal.GetIteratorBegin(),
                           this->InputPortal.GetIteratorEnd(),
                           this->ValuesPortal.Get(index));

      dax::Id resultIndex =
          static_cast<dax::Id>(
            std::distance(this->InputPortal.GetIteratorBegin(), resultPos));
      this->OutputPortal.Set(index, resultIndex);
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

  template<class InputPortalType,class ValuesPortalType,class OutputPortalType,class Compare>
  struct LowerBoundsComparisonKernel {
    InputPortalType InputPortal;
    ValuesPortalType ValuesPortal;
    OutputPortalType OutputPortal;
    Compare CompareFunctor;

    DAX_CONT_EXPORT
    LowerBoundsComparisonKernel(InputPortalType inputPortal,
                      ValuesPortalType valuesPortal,
                      OutputPortalType outputPortal,
                      Compare comp)
      : InputPortal(inputPortal),
        ValuesPortal(valuesPortal),
        OutputPortal(outputPortal),
        CompareFunctor(comp) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      // This method assumes that (1) InputPortalType can return working
      // iterators in the execution environment and that (2) methods not
      // specified with DAX_EXEC_EXPORT (such as the STL algorithms) can be
      // called from the execution environment. Neither one of these is
      // necessarily true, but it is true for the current uses of this general
      // function and I don't want to compete with STL if I don't have to.

      typename InputPortalType::IteratorType resultPos =
          std::lower_bound(this->InputPortal.GetIteratorBegin(),
                           this->InputPortal.GetIteratorEnd(),
                           this->ValuesPortal.Get(index),
                           this->CompareFunctor);

      dax::Id resultIndex =
          static_cast<dax::Id>(
            std::distance(this->InputPortal.GetIteratorBegin(), resultPos));
      this->OutputPortal.Set(index, resultIndex);
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };


public:
  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag> &values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &output)
  {
    dax::Id arraySize = values.GetNumberOfValues();

    LowerBoundsKernel<
        typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>::PortalExecution>
        kernel(input.PrepareForInput(),
               values.PrepareForInput(),
               output.PrepareForOutput(arraySize));

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template<typename T, class CIn, class CVal, class COut, class Compare>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag> &values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &output,
      Compare comp)
  {
    dax::Id arraySize = values.GetNumberOfValues();

    LowerBoundsComparisonKernel<
        typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag>::PortalExecution,
        Compare>
        kernel(input.PrepareForInput(),
               values.PrepareForInput(),
               output.PrepareForOutput(arraySize),
               comp);

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    DeviceAdapterAlgorithmGeneral<DerivedAlgorithm,DeviceAdapterTag>::
        LowerBounds(input, values_output, values_output);
  }

  //--------------------------------------------------------------------------
  // Scan Exclusive
private:
  template<typename PortalType>
  struct SetConstantKernel
  {
    typedef typename PortalType::ValueType ValueType;
    PortalType Portal;
    ValueType Value;

    DAX_CONT_EXPORT
    SetConstantKernel(const PortalType &portal, ValueType value)
      : Portal(portal), Value(value) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const
    {
      this->Portal.Set(index, this->Value);
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    typedef dax::cont::ArrayHandle<
        T,dax::cont::ArrayContainerControlTagBasic,DeviceAdapterTag>
        TempArrayType;
    typedef dax::cont::ArrayHandle<T,COut,DeviceAdapterTag> OutputArrayType;

    TempArrayType inclusiveScan;
    T result = DerivedAlgorithm::ScanInclusive(input, inclusiveScan);

    dax::Id numValues = inclusiveScan.GetNumberOfValues();
    if (numValues < 1)
      {
      return result;
      }

    typedef typename TempArrayType::PortalConstExecution SrcPortalType;
    SrcPortalType srcPortal = inclusiveScan.PrepareForInput();

    typedef typename OutputArrayType::PortalExecution DestPortalType;
    DestPortalType destPortal = output.PrepareForOutput(numValues);

    // Set first value in output (always 0).
    DerivedAlgorithm::Schedule(
          SetConstantKernel<DestPortalType>(destPortal,0), 1);
    // Shift remaining values over by one.
    DerivedAlgorithm::Schedule(
          CopyKernel<SrcPortalType,DestPortalType>(srcPortal,
                                                   destPortal,
                                                   0,
                                                   1),
          numValues - 1);

    return result;
  }

  //--------------------------------------------------------------------------
  // Scan Inclusive
private:
  template<typename PortalType>
  struct ScanKernel : dax::exec::internal::WorkletBase
  {
    PortalType Portal;
    dax::Id Stride;
    dax::Id Offset;
    dax::Id Distance;

    DAX_CONT_EXPORT
    ScanKernel(const PortalType &portal, dax::Id stride, dax::Id offset)
      : Portal(portal),
        Stride(stride),
        Offset(offset),
        Distance(stride/2)
    {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const
    {
      typedef typename PortalType::ValueType ValueType;

      dax::Id leftIndex = this->Offset + index*this->Stride;
      dax::Id rightIndex = leftIndex + this->Distance;

      DAX_ASSERT_EXEC(leftIndex < this->Portal.GetNumberOfValues(), *this);

      if (rightIndex < this->Portal.GetNumberOfValues())
        {
        ValueType leftValue = this->Portal.Get(leftIndex);
        ValueType rightValue = this->Portal.Get(rightIndex);
        this->Portal.Set(rightIndex, leftValue+rightValue);
        }
    }
  };

public:
  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanInclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    typedef typename
        dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>::PortalExecution
            PortalType;

    DerivedAlgorithm::Copy(input, output);

    dax::Id numValues = output.GetNumberOfValues();
    if (numValues < 1)
      {
      return 0;
      }

    PortalType portal = output.PrepareForInPlace();

    dax::Id stride;
    for (stride = 2; stride-1 < numValues; stride *= 2)
      {
      ScanKernel<PortalType> kernel(portal, stride, stride/2 - 1);
      DerivedAlgorithm::Schedule(kernel, numValues/stride);
      }

    // Do reverse operation on odd indices. Start at stride we were just at.
    for (stride /= 2; stride > 1; stride /= 2)
      {
      ScanKernel<PortalType> kernel(portal, stride, stride - 1);
      DerivedAlgorithm::Schedule(kernel, numValues/stride);
      }

    return GetExecutionValue(output, numValues-1);
  }

  //--------------------------------------------------------------------------
  // Sort
private:
  template<typename PortalType, typename CompareType>
  struct BitonicSortMergeKernel : dax::exec::internal::WorkletBase
  {
    PortalType Portal;
    CompareType Compare;
    dax::Id GroupSize;

    DAX_CONT_EXPORT
    BitonicSortMergeKernel(const PortalType &portal,
                           const CompareType &compare,
                           dax::Id groupSize)
      : Portal(portal), Compare(compare), GroupSize(groupSize) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      typedef typename PortalType::ValueType ValueType;

      dax::Id groupIndex = index%this->GroupSize;
      dax::Id blockSize = 2*this->GroupSize;
      dax::Id blockIndex = index/this->GroupSize;

      dax::Id lowIndex = blockIndex * blockSize + groupIndex;
      dax::Id highIndex = lowIndex + this->GroupSize;

      if (highIndex < this->Portal.GetNumberOfValues())
        {
        ValueType lowValue = this->Portal.Get(lowIndex);
        ValueType highValue = this->Portal.Get(highIndex);
        if (this->Compare(highValue, lowValue))
          {
          this->Portal.Set(highIndex, lowValue);
          this->Portal.Set(lowIndex, highValue);
          }
        }
    }
  };

  template<typename PortalType, typename CompareType>
  struct BitonicSortCrossoverKernel : dax::exec::internal::WorkletBase
  {
    PortalType Portal;
    CompareType Compare;
    dax::Id GroupSize;

    DAX_CONT_EXPORT
    BitonicSortCrossoverKernel(const PortalType &portal,
                               const CompareType &compare,
                               dax::Id groupSize)
      : Portal(portal), Compare(compare), GroupSize(groupSize) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      typedef typename PortalType::ValueType ValueType;

      dax::Id groupIndex = index%this->GroupSize;
      dax::Id blockSize = 2*this->GroupSize;
      dax::Id blockIndex = index/this->GroupSize;

      dax::Id lowIndex = blockIndex*blockSize + groupIndex;
      dax::Id highIndex = blockIndex*blockSize + (blockSize - groupIndex - 1);

      if (highIndex < this->Portal.GetNumberOfValues())
        {
        ValueType lowValue = this->Portal.Get(lowIndex);
        ValueType highValue = this->Portal.Get(highIndex);
        if (this->Compare(highValue, lowValue))
          {
          this->Portal.Set(highIndex, lowValue);
          this->Portal.Set(lowIndex, highValue);
          }
        }
    }
  };

  struct DefaultCompareFunctor
  {

    template<typename T>
    DAX_EXEC_EXPORT
    bool operator()(const T& first, const T& second) const
    {
      return first < second;
    }
  };

public:
  template<typename T, class Container, class CompareType>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values,
      CompareType compare)
  {
    typedef typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>
        ArrayType;
    typedef typename ArrayType::PortalExecution PortalType;

    dax::Id numValues = values.GetNumberOfValues();
    if (numValues < 2) { return; }

    PortalType portal = values.PrepareForInPlace();

    dax::Id numThreads = 1;
    while (numThreads < numValues) { numThreads *= 2; }
    numThreads /= 2;

    typedef BitonicSortMergeKernel<PortalType,CompareType> MergeKernel;
    typedef BitonicSortCrossoverKernel<PortalType,CompareType> CrossoverKernel;

    for (dax::Id crossoverSize = 1;
         crossoverSize < numValues;
         crossoverSize *= 2)
      {
      DerivedAlgorithm::Schedule(CrossoverKernel(portal,compare,crossoverSize),
                                 numThreads);
      for (dax::Id mergeSize = crossoverSize/2; mergeSize > 0; mergeSize /= 2)
        {
        DerivedAlgorithm::Schedule(MergeKernel(portal,compare,mergeSize),
                                   numThreads);
        }
      }
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values)
  {
    DerivedAlgorithm::Sort(values, DefaultCompareFunctor());
  }

  //--------------------------------------------------------------------------
  // Sort by Key
private:
  template<typename T, typename U, class Compare=DefaultCompareFunctor>
  struct KeyCompare
  {
    KeyCompare(): CompareFunctor() {}
    explicit KeyCompare(Compare c): CompareFunctor(c) {}

    DAX_EXEC_EXPORT
    bool operator()(const dax::Pair<T,U>& a, const dax::Pair<T,U>& b) const
    {
      return CompareFunctor(a.first,b.first);
    }
  private:
    Compare CompareFunctor;
  };

public:

  template<typename T, typename U, class ContainerT,  class ContainerU>
  DAX_CONT_EXPORT static void SortByKey(
      dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag> &keys,
      dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag> &values)
  {
    //combine the keys and values into a ZipArrayHandle
    //we than need to specify a custom compare function wrapper
    //that only checks for key side of the pair, using a custom compare functor.
    typedef dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag> KeyType;
    typedef dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag> ValueType;
    typedef dax::cont::internal::ArrayHandleZip<KeyType,ValueType> ZipHandleType;
    typedef typename ZipHandleType::Superclass HandleType;

    //slice the zip handle so we can pass it to the sort algorithm
    HandleType zipHandle =
                    dax::cont::internal::make_ArrayHandleZip(keys,values);
    DerivedAlgorithm::Sort(zipHandle,KeyCompare<T,U>());
  }

  template<typename T, typename U, class ContainerT,  class ContainerU, class Compare>
  DAX_CONT_EXPORT static void SortByKey(
      dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag> &keys,
      dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag> &values,
      Compare comp)
  {
    //combine the keys and values into a ZipArrayHandle
    //we than need to specify a custom compare function wrapper
    //that only checks for key side of the pair, using the custom compare
    //functor that the user passed in
    typedef dax::cont::ArrayHandle<T,ContainerT,DeviceAdapterTag> KeyType;
    typedef dax::cont::ArrayHandle<U,ContainerU,DeviceAdapterTag> ValueType;
    typedef dax::cont::internal::ArrayHandleZip<KeyType,ValueType> ZipHandleType;
    typedef typename ZipHandleType::Superclass HandleType;

    //slice the zip handle so we can pass it to the sort algorithm
    HandleType zipHandle =
                    dax::cont::internal::make_ArrayHandleZip(keys,values);
    DerivedAlgorithm::Sort(zipHandle,KeyCompare<T,U,Compare>(comp));
  }

  //--------------------------------------------------------------------------
  // Stream Compact
private:
  template<class StencilPortalType, class OutputPortalType>
  struct StencilToIndexFlagKernel
  {
    typedef typename StencilPortalType::ValueType StencilValueType;
    StencilPortalType StencilPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    StencilToIndexFlagKernel(StencilPortalType stencilPortal,
                             OutputPortalType outputPortal)
      : StencilPortal(stencilPortal), OutputPortal(outputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const
    {
      StencilValueType value = this->StencilPortal.Get(index);
      bool flag = dax::not_default_constructor<StencilValueType>()(value);
      this->OutputPortal.Set(index, flag ? 1 : 0);
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

  template<class InputPortalType,
           class StencilPortalType,
           class IndexPortalType,
           class OutputPortalType>
  struct CopyIfKernel
  {
    InputPortalType InputPortal;
    StencilPortalType StencilPortal;
    IndexPortalType IndexPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    CopyIfKernel(InputPortalType inputPortal,
                 StencilPortalType stencilPortal,
                 IndexPortalType indexPortal,
                 OutputPortalType outputPortal)
      : InputPortal(inputPortal),
        StencilPortal(stencilPortal),
        IndexPortal(indexPortal),
        OutputPortal(outputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const
    {
      typedef typename StencilPortalType::ValueType StencilValueType;
      StencilValueType stencilValue = this->StencilPortal.Get(index);
      if (dax::not_default_constructor<StencilValueType>()(stencilValue))
        {
        dax::Id outputIndex = this->IndexPortal.Get(index);

        typedef typename OutputPortalType::ValueType OutputValueType;
        OutputValueType value = this->InputPortal.Get(index);

        this->OutputPortal.Set(outputIndex, value);
        }
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

public:

  template<typename T, typename U, class CIn, class CStencil, class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>& input,
      const dax::cont::ArrayHandle<U,CStencil,DeviceAdapterTag>& stencil,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>& output)
  {
    DAX_ASSERT_CONT(input.GetNumberOfValues() == stencil.GetNumberOfValues());
    dax::Id arrayLength = stencil.GetNumberOfValues();

    typedef dax::cont::ArrayHandle<
        dax::Id, dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>
        IndexArrayType;
    IndexArrayType indices;

    typedef typename dax::cont::ArrayHandle<U,CStencil,DeviceAdapterTag>
        ::PortalConstExecution StencilPortalType;
    StencilPortalType stencilPortal = stencil.PrepareForInput();

    typedef typename IndexArrayType::PortalExecution IndexPortalType;
    IndexPortalType indexPortal = indices.PrepareForOutput(arrayLength);

    StencilToIndexFlagKernel<
        StencilPortalType, IndexPortalType> indexKernel(stencilPortal,
                                                        indexPortal);

    DerivedAlgorithm::Schedule(indexKernel, arrayLength);

    dax::Id outArrayLength = DerivedAlgorithm::ScanExclusive(indices, indices);

    typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>
        ::PortalConstExecution InputPortalType;
    InputPortalType inputPortal = input.PrepareForInput();

    typedef typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>
        ::PortalExecution OutputPortalType;
    OutputPortalType outputPortal = output.PrepareForOutput(outArrayLength);

    CopyIfKernel<
        InputPortalType,
        StencilPortalType,
        IndexPortalType,
        OutputPortalType>copyKernel(inputPortal,
                                    stencilPortal,
                                    indexPortal,
                                    outputPortal);
    DerivedAlgorithm::Schedule(copyKernel, arrayLength);
  }

  template<typename T, class CStencil, class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag> &stencil,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &output)
  {
    typedef dax::cont::ArrayHandle< dax::Id,
                    dax::cont::internal::ArrayContainerControlTagCounting,
                    DeviceAdapterTag> CountingHandleType;
    typedef dax::cont::internal::ArrayPortalCounting<dax::Id> CountingPortal;

    CountingHandleType input( CountingPortal(0,stencil.GetNumberOfValues()) );
    DerivedAlgorithm::StreamCompact(input, stencil, output);
  }

  //--------------------------------------------------------------------------
  // Unique
private:
  template<class InputPortalType, class StencilPortalType>
  struct ClassifyUniqueKernel {
    InputPortalType InputPortal;
    StencilPortalType StencilPortal;

    DAX_CONT_EXPORT
    ClassifyUniqueKernel(InputPortalType inputPortal,
                         StencilPortalType stencilPortal)
      : InputPortal(inputPortal), StencilPortal(stencilPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      typedef typename StencilPortalType::ValueType ValueType;
      if (index == 0)
        {
        // Always copy first value.
        this->StencilPortal.Set(index, ValueType(1));
        }
      else
        {
        ValueType flag = ValueType(this->InputPortal.Get(index-1)
                                   != this->InputPortal.Get(index));
        this->StencilPortal.Set(index, flag);
        }
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

  template<class InputPortalType, class StencilPortalType, class Compare>
  struct ClassifyUniqueComparisonKernel {
    InputPortalType InputPortal;
    StencilPortalType StencilPortal;
    Compare CompareFunctor;

    DAX_CONT_EXPORT
    ClassifyUniqueComparisonKernel(InputPortalType inputPortal,
                         StencilPortalType stencilPortal,
                         Compare comp):
      InputPortal(inputPortal),
      StencilPortal(stencilPortal),
      CompareFunctor(comp) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      typedef typename StencilPortalType::ValueType ValueType;
      if (index == 0)
        {
        // Always copy first value.
        this->StencilPortal.Set(index, ValueType(1));
        }
      else
        {
        //comparison predicate returns true when they match
        const bool same = !(this->CompareFunctor(this->InputPortal.Get(index-1),
                                                 this->InputPortal.Get(index)));
        ValueType flag = ValueType(same);
        this->StencilPortal.Set(index, flag);
        }
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

public:
  template<typename T, class Container>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values)
  {
    dax::cont::ArrayHandle<
        dax::Id, dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>
        stencilArray;
    dax::Id inputSize = values.GetNumberOfValues();

    ClassifyUniqueKernel<
        typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<dax::Id,dax::cont::ArrayContainerControlTagBasic,DeviceAdapterTag>::PortalExecution>
        classifyKernel(values.PrepareForInput(),
                       stencilArray.PrepareForOutput(inputSize));
    DerivedAlgorithm::Schedule(classifyKernel, inputSize);

    dax::cont::ArrayHandle<
        T, dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>
        outputArray;

    DerivedAlgorithm::StreamCompact(values, stencilArray, outputArray);

    DerivedAlgorithm::Copy(outputArray, values);
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &values,
      Compare comp)
  {
    dax::cont::ArrayHandle<
        dax::Id, dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>
        stencilArray;
    dax::Id inputSize = values.GetNumberOfValues();

    ClassifyUniqueComparisonKernel<
        typename dax::cont::ArrayHandle<T,Container,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<dax::Id,dax::cont::ArrayContainerControlTagBasic,DeviceAdapterTag>::PortalExecution,
        Compare>
        classifyKernel(values.PrepareForInput(),
                       stencilArray.PrepareForOutput(inputSize),
                       comp);
    DerivedAlgorithm::Schedule(classifyKernel, inputSize);

    dax::cont::ArrayHandle<
        T, dax::cont::ArrayContainerControlTagBasic, DeviceAdapterTag>
        outputArray;

    DerivedAlgorithm::StreamCompact(values, stencilArray, outputArray);

    DerivedAlgorithm::Copy(outputArray, values);
  }

  //--------------------------------------------------------------------------
  // Upper bounds
private:
  template<class InputPortalType,class ValuesPortalType,class OutputPortalType>
  struct UpperBoundsKernel {
    InputPortalType InputPortal;
    ValuesPortalType ValuesPortal;
    OutputPortalType OutputPortal;

    DAX_CONT_EXPORT
    UpperBoundsKernel(InputPortalType inputPortal,
                      ValuesPortalType valuesPortal,
                      OutputPortalType outputPortal)
      : InputPortal(inputPortal),
        ValuesPortal(valuesPortal),
        OutputPortal(outputPortal) {  }

    DAX_EXEC_EXPORT
    void operator()(dax::Id index) const {
      // This method assumes that (1) InputPortalType can return working
      // iterators in the execution environment and that (2) methods not
      // specified with DAX_EXEC_EXPORT (such as the STL algorithms) can be
      // called from the execution environment. Neither one of these is
      // necessarily true, but it is true for the current uses of this general
      // function and I don't want to compete with STL if I don't have to.

      typename InputPortalType::IteratorType resultPos =
          std::upper_bound(this->InputPortal.GetIteratorBegin(),
                           this->InputPortal.GetIteratorEnd(),
                           this->ValuesPortal.Get(index));

      dax::Id resultIndex =
          static_cast<dax::Id>(
            std::distance(this->InputPortal.GetIteratorBegin(), resultPos));
      this->OutputPortal.Set(index, resultIndex);
    }

    DAX_CONT_EXPORT
    void SetErrorMessageBuffer(const dax::exec::internal::ErrorMessageBuffer &)
    {  }
  };

public:
  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &input,
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag> &values,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &output)
  {
    dax::Id arraySize = values.GetNumberOfValues();

    UpperBoundsKernel<
        typename dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag>::PortalConstExecution,
        typename dax::cont::ArrayHandle<T,COut,DeviceAdapterTag>::PortalExecution>
        kernel(input.PrepareForInput(),
               values.PrepareForInput(),
               output.PrepareForOutput(arraySize));

    DerivedAlgorithm::Schedule(kernel, arraySize);
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void UpperBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    DeviceAdapterAlgorithmGeneral<DerivedAlgorithm,DeviceAdapterTag>::
        UpperBounds(input, values_output, values_output);
  }

};


}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithmGeneral_h
