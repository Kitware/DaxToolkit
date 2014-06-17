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
#ifndef __VerifyThresholdTopology_worklet_
#define __VerifyThresholdTopology_worklet_

#include <dax/exec/WorkletGenerateTopology.h>
#include <dax/exec/Assert.h>

namespace dax {
namespace worklet {
namespace testing {

struct VerifyThresholdTopology : public dax::exec::WorkletGenerateTopology
{
  typedef void ControlSignature(TopologyIn, TopologyOut,FieldIn);
  typedef void ExecutionSignature(Vertices(_1), Vertices(_2), _3, VisitIndex);

  template<typename InputCellTag, typename OutputCellTag, typename T>
  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<InputCellTag> &inVertices,
                  dax::exec::CellVertices<OutputCellTag> &outVertices,
                  const T&,
                  const dax::Id& visit_index) const
  {
    DAX_ASSERT_EXEC(visit_index==0, *this);
    outVertices.SetFromTuple(inVertices.GetAsTuple());
  }
};

}
}
}

#endif