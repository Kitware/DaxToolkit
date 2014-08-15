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

#ifndef __dax_exec_internal_kernel_VisitIndexWorklet_h
#define __dax_exec_internal_kernel_VisitIndexWorklet_h

#include <dax/Types.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/internal/WorkletSignatureFunctions.h>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class Worklet, class ModifiedWorkletSignatures>
class DerivedWorklet : public Worklet
{
public:
  DerivedWorklet(const Worklet& worklet):
    Worklet(worklet)
    {}

  //we have to build the new worklet signatures, we explicitly pass in
  //ModifiedWorkletSignatures instead of the expanded Control and Execution
  //signatures to work around VisualStudio + NVCC pre-processor bugs. Basically
  //nvcc creates a typedef with the following snippet <(void<(int)1>)>, which
  //makes visual studio have a parse error
  typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ControlSignature>::type ControlSignature;
   typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ExecutionSignature>::type ExecutionSignature;
};


struct ComputeVisitIndex : public WorkletMapField
{
  typedef void ControlSignature(FieldIn,FieldOut);
  typedef _2 ExecutionSignature(_1,WorkId);

  DAX_EXEC_EXPORT dax::Id operator()(const dax::Id& LowerBoundsCount,
                                     const dax::Id& workId) const
  {
    return workId - LowerBoundsCount;
  }
};

}
}
}
} //dax::exec::internal::kernel


#endif // __dax_exec_internal_kernel_VisitIndexWorklet_h
