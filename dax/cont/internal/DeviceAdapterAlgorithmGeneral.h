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

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/Functional.h>

#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <dax/tbb/cont/internal/DeviceAdapterTagTBB.h>

namespace dax {
namespace cont {
namespace internal {

/// \brief
///
/// This struct provides algorithms that implement "general" device adapter
/// algorithms. If a device adapter provides implementations for Schedule,
/// Sort, and Scan, the rest of the algorithms can be implemented by calling
/// these functions. An easy way to implement the DeviceAdapterAlgorithm
/// specialization is to subclass this and override the implementation of
/// methods as necessary.
///
template<class DeviceAdapterTag>
struct DeviceAdapterAlgorithmGeneral
{

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static void Copy(
      const dax::cont::ArrayHandle<T, CIn, DeviceAdapterTag> &daxNotUsed(input),
      dax::cont::ArrayHandle<T, COut, DeviceAdapterTag> &daxNotUsed(output))
  {
    //TODO
  }

  template<typename T, class CIn, class CVal, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTag> &daxNotUsed(input),
      const dax::cont::ArrayHandle<T,CVal,DeviceAdapterTag> &daxNotUsed(values),
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &daxNotUsed(output))
  {
    //TODO
  }

  template<class CIn, class COut>
  DAX_CONT_EXPORT static void LowerBounds(
      const dax::cont::ArrayHandle<dax::Id,CIn,DeviceAdapterTag> &input,
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &values_output)
  {
    DeviceAdapterAlgorithmGeneral<DeviceAdapterTag>::
        LowerBounds(input, values_output, values_output);
  }

private:
  typedef dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>
      Algorithm;

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
  /// An implementation of StreamCompact that works on any DeviceAdapter that
  /// defines Schedule and ScanExclusive.
  ///
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

    Algorithm::Schedule(indexKernel, arrayLength);

    dax::Id outArrayLength = Algorithm::ScanExclusive(indices, indices);

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
    Algorithm::Schedule(copyKernel, arrayLength);
  }

  template<typename T, class CStencil, class COut>
  DAX_CONT_EXPORT static void StreamCompact(
      const dax::cont::ArrayHandle<T,CStencil,DeviceAdapterTag> &daxNotUsed(stencil),
      dax::cont::ArrayHandle<dax::Id,COut,DeviceAdapterTag> &daxNotUsed(output))
  {
    //TODO
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Unique(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTag> &daxNotUsed(values))
  {
    //TODO
  }

};


}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_DeviceAdapterAlgorithmGeneral_h
