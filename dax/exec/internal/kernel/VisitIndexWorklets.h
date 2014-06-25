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

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class Worklet, typename CSig, typename ESig>
class DerivedWorklet : public Worklet
{
public:
  DerivedWorklet(const Worklet& worklet):
    Worklet(worklet)
    {}
  typedef CSig ControlSignature;
  typedef ESig ExecutionSignature;
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
