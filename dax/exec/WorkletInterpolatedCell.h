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
#ifndef __dax_exec_WorkletInterpolatedCell_h
#define __dax_exec_WorkletInterpolatedCell_h

#include <dax/exec/internal/WorkletBase.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/Geometry.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Superclass for worklets that generate new coordinates. Use this when the worklet
/// needs to create new coordinates.
///
class WorkletInterpolatedCell : public dax::exec::internal::WorkletBase
{
public:
  typedef dax::cont::sig::PermutedCell DomainType;

  DAX_EXEC_EXPORT WorkletInterpolatedCell() { }
protected:
  typedef dax::cont::sig::Cell Cell;
  typedef dax::cont::arg::Field Field;
  typedef dax::cont::sig::Point Point;
  typedef dax::cont::arg::Geometry Geometry;
  typedef dax::cont::arg::Topology Topology;
  typedef dax::cont::arg::Topology::Vertices Vertices;

  typedef dax::cont::sig::VisitIndex VisitIndex;
};

}
}

#endif //__dax_exec_WorkletGenerateCells_h
