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

#ifndef __dax_exec_internal_kernel_GenerateWorklets_h
#define __dax_exec_internal_kernel_GenerateWorklets_h

#include <dax/Types.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/math/VectorAnalysis.h>

#include <dax/cont/internal/EdgeInterpolatedGrid.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

struct ClearUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT void operator()(T &t) const
  {
    t = static_cast<T>(0);
  }
};

struct Index : public WorkletMapField
{
  typedef void ControlSignature(FieldOut);
  typedef _1 ExecutionSignature(WorkId);

  DAX_EXEC_EXPORT dax::Id operator()(dax::Id index) const
  {
    return index;
  }
};

struct GetUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(FieldOut);
  typedef void ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT void operator()(T& t) const
  {
    t = static_cast<T>(1);
  }
};

template<class InPortalType,
         class InterpolationPortalType,
         class OutPortalType >
struct InterpolateFieldToField
  {
    DAX_CONT_EXPORT InterpolateFieldToField(const InPortalType &inPortal,
                            const InterpolationPortalType & interpPortal,
                                          const OutPortalType &outPortal) :
    Input(inPortal),
    Interpolation(interpPortal),
    Output(outPortal)
    {  }


    DAX_EXEC_EXPORT void operator()(dax::Id index) const
    {
      const dax::PointAsEdgeInterpolation interpolationInfo = this->Interpolation.Get(index);


      typedef typename InPortalType::ValueType InValueType;
      typedef typename OutPortalType::ValueType OutValueType;

      const InValueType first = this->Input.Get(interpolationInfo.EdgeIdFirst);
      const InValueType second = this->Input.Get(interpolationInfo.EdgeIdSecond);

      this->Output.Set(index,
                       dax::math::Lerp(first,second,interpolationInfo.Weight) );
    }

    DAX_CONT_EXPORT void SetErrorMessageBuffer(
        const dax::exec::internal::ErrorMessageBuffer &) {  }

    InPortalType Input;
    InterpolationPortalType Interpolation;
    OutPortalType Output;
  };

template< typename ReductionMapType >
struct Offset2CountFunctor : dax::exec::internal::WorkletBase
{
  typename ReductionMapType::PortalConstExecution OffsetsPortal;
  typename ReductionMapType::PortalExecution CountsPortal;
  dax::Id MaxId;
  dax::Id OffsetEnd;

  Offset2CountFunctor(
      typename ReductionMapType::PortalConstExecution offsetsPortal,
      typename ReductionMapType::PortalExecution countsPortal,
      dax::Id maxId,
      dax::Id offsetEnd)
    : OffsetsPortal(offsetsPortal),
      CountsPortal(countsPortal),
      MaxId(maxId),
      OffsetEnd(offsetEnd) {  }

  DAX_EXEC_EXPORT
  void operator()(dax::Id index) const {
    dax::Id thisOffset = this->OffsetsPortal.Get(index);
    dax::Id nextOffset;
    if (index == this->MaxId)
      {
      nextOffset = this->OffsetEnd;
      }
    else
      {
      nextOffset = this->OffsetsPortal.Get(index+1);
      }
    this->CountsPortal.Set(index, nextOffset - thisOffset);
  }
};

}
}
}
} //dax::exec::internal::kernel


#endif // __dax_exec_internal_kernel_GenerateWorklets_h
