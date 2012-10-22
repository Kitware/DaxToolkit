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

// This header file provides algorithms that implement "general" device adapter
// algorithms. If a device adapter provides implementations for Schedule, Sort,
// and Scan, the rest of the algorithms can be implemented by calling these
// functions.

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/Functional.h>

#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>

namespace dax {
namespace cont {
namespace internal {
//template<class FunctorType>
//DAX_CONT_EXPORT void Schedule(FunctorType functor,
//                              dax::Id numInstances,
//                              dax::tbb::cont::DeviceAdapterTagTBB);
//template<typename T, class CIn, class COut>
//DAX_CONT_EXPORT T ExclusiveScan(
//    const dax::cont::ArrayHandle<T,CIn,dax::tbb::cont::DeviceAdapterTagTBB>
//        &input,
//    dax::cont::ArrayHandle<T,COut,dax::tbb::cont::DeviceAdapterTagTBB>
//        &output,
//    dax::tbb::cont::DeviceAdapterTagTBB);

namespace detail {

template<class StencilPortalType, class OutputPortalType>
struct StreamCompactGeneralStencilToIndexFlag
{
  typedef typename StencilPortalType::ValueType StencilValueType;
  StencilPortalType StencilPortal;
  OutputPortalType OutputPortal;

  DAX_CONT_EXPORT
  StreamCompactGeneralStencilToIndexFlag(StencilPortalType stencilPortal,
                                         OutputPortalType outputPortal)
    : StencilPortal(stencilPortal), OutputPortal(outputPortal) {  }

  DAX_EXEC_EXPORT
  void operator()(dax::Id index)
  {
    StencilValueType value = this->StencilPortal.Get(index);
    bool flag = dax::not_default_constructor<StencilValueType>()(value);
    this->OutputPortal.Set(index, flag ? 1 : 0);
  }
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
  void operator()(dax::Id index)
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
};

} // namespace detail

/// An implementation of StreamCompact that works on any DeviceAdapter that
/// defines Schedule and ExclusiveScan.
template<typename T,
         typename U,
         class CIn,
         class CStencil,
         class COut,
         class DeviceAdapter>
DAX_CONT_EXPORT void StreamCompactGeneral(
    const dax::cont::ArrayHandle<T,CIn,DeviceAdapter>& input,
    const dax::cont::ArrayHandle<U,CStencil,DeviceAdapter>& stencil,
    dax::cont::ArrayHandle<T,COut,DeviceAdapter>& output,
    DeviceAdapter)
{
  DAX_ASSERT_CONT(input.GetNumberOfValues() == stencil.GetNumberOfValues());
  dax::Id arrayLength = stencil.GetNumberOfValues();

  typedef dax::cont::ArrayHandle<
      dax::Id, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      IndexArrayType;
  IndexArrayType indices;

  typedef typename dax::cont::ArrayHandle<U,CStencil,DeviceAdapter>
      ::PortalConstExecution StencilPortalType;
  StencilPortalType stencilPortal = stencil.PrepareForInput();

  typedef typename IndexArrayType::PortalExecution IndexPortalType;
  IndexPortalType indexPortal = indices.PrepareForOutput(arrayLength);

  detail::StreamCompactGeneralStencilToIndexFlag<
      StencilPortalType, IndexPortalType> indexKernel(stencilPortal,
                                                      indexPortal);

  dax::cont::internal::Schedule(indexKernel, arrayLength, DeviceAdapter());

  dax::Id outArrayLength =
      dax::cont::internal::ExclusiveScan(indices, indices, DeviceAdapter());

  typedef typename dax::cont::ArrayHandle<T,CIn,DeviceAdapter>
      ::PortalConstExecution InputPortalType;
  InputPortalType inputPortal = input.PrepareForInput();

  typedef typename dax::cont::ArrayHandle<T,COut,DeviceAdapter>
      ::PortalExecution OutputPortalType;
  OutputPortalType outputPortal = output.PrepareForOutput(outArrayLength);

  detail::CopyIfKernel<
      InputPortalType,
      StencilPortalType,
      IndexPortalType,
      OutputPortalType>copyKernel(inputPortal,
                                  stencilPortal,
                                  indexPortal,
                                  outputPortal);
  dax::cont::internal::Schedule(copyKernel, outArrayLength, DeviceAdapter());
}


}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithmGeneral_h
