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

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

struct ClearUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef void ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT void operator()(T &t) const
  {
    t = static_cast<T>(0);
  }
};

struct Index : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef _1 ExecutionSignature(WorkId);

  DAX_EXEC_EXPORT dax::Id operator()(dax::Id index) const
  {
    return index;
  }
};

struct GetUsedPointsFunctor : public WorkletMapField
{
  typedef void ControlSignature(Field(Out));
  typedef void ExecutionSignature(_1);

  template<typename T>
  DAX_EXEC_EXPORT void operator()(T& t) const
  {
    t = static_cast<T>(1);
  }
};

template<class InVec3PortalType, class OutVec3PortalType>
struct InterpolateEdgesToPoint
  {
    DAX_CONT_EXPORT InterpolateEdgesToPoint(const InVec3PortalType &coords,
                                            const OutVec3PortalType &interpCoords) :
    Coords(coords),
    InterpCoords(interpCoords)
    {  }


    DAX_EXEC_EXPORT void operator()(dax::Id index) const
    {
      const dax::Vector3 pointInterpInfo = InterpCoords.Get(index);
      const dax::Vector3 point1 = Coords.Get(pointInterpInfo[0]);
      const dax::Vector3 point2 = Coords.Get(pointInterpInfo[1]);
      InterpCoords.Set(index, dax::math::Lerp(point1,point2,pointInterpInfo[2]));
    }

    DAX_CONT_EXPORT void SetErrorMessageBuffer(
        const dax::exec::internal::ErrorMessageBuffer &) {  }

    InVec3PortalType Coords;
    OutVec3PortalType InterpCoords;
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
