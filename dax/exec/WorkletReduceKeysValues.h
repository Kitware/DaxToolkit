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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_WorkletReduceKeysValues_h
#define __dax_exec_WorkletReduceKeysValues_h

#include <dax/exec/internal/WorkletBase.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/ReductionCount.h>
#include <dax/cont/sig/KeyGroup.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Superclass for worklets that generate new coordinates. Use this when the worklet
/// needs to create new coordinates.
///
class WorkletReduceKeysValues : public dax::exec::internal::WorkletBase
{
public:
  typedef dax::cont::sig::PermutedAnyDomain DomainType;

  DAX_EXEC_EXPORT WorkletReduceKeysValues() { }
protected:
  typedef dax::cont::arg::Field Values;

  typedef dax::cont::sig::ReductionCount ReductionCount;
  typedef dax::cont::sig::ReductionOffset ReductionOffset;
  typedef dax::cont::sig::KeyGroup KeyGroup;
};

}
}

#endif //__dax_exec_WorkletGenerateCells_h
