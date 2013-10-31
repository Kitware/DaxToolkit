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
#ifndef __dax_exec_WorkletGenerateKeysValues_h
#define __dax_exec_WorkletGenerateKeysValues_h

#include <dax/exec/internal/WorkletBase.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------

/// Superclass for worklets that generate keys and/or values for each input
/// cell.  This functionality is most often used to connect components based on
/// input topology (such as accumulate all values associated with a point or
/// edge).
///
class WorkletGenerateKeysValues : public dax::exec::internal::WorkletBase
{
public:
  typedef dax::cont::sig::PermutedCell DomainType;

  DAX_EXEC_EXPORT WorkletGenerateKeysValues() { }
protected:
  typedef dax::cont::arg::Field Field;
  typedef dax::cont::arg::Topology Topology;
  typedef dax::cont::arg::Topology::Vertices Vertices;
  typedef dax::cont::sig::Cell Cell;
  typedef dax::cont::sig::Point Point;
  typedef dax::cont::sig::VisitIndex VisitIndex;
};

}
}

#endif //__dax_exec_WorkletGenerateKeysValues_h
