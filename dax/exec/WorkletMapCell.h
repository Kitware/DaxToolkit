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
#ifndef __dax_exec_WorkletMapCell_h
#define __dax_exec_WorkletMapCell_h

#include <dax/exec/internal/WorkletBase.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/arg/Bind.h>
#include <dax/exec/arg/BindCellPoints.h>

#include <boost/mpl/if.hpp>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Superclass for worklets that map points to cell. Use this when the worklet
/// needs "CellArray" information i.e. information about what points form a
/// cell.
///
class WorkletMapCell : public dax::exec::internal::WorkletBase
{
public:
  typedef WorkletMapCell WorkType;
  typedef dax::cont::sig::Cell DomainType;

  DAX_EXEC_EXPORT WorkletMapCell() { }
protected:
  typedef dax::cont::arg::Field Field;
  typedef dax::cont::arg::Topology Topology;
  typedef dax::cont::sig::Point Point;
  typedef dax::cont::sig::Cell Cell;
};

namespace arg {

template <int N, typename Invocation>
class Bind<WorkletMapCell, dax::cont::sig::Arg<N>, Invocation>
{
  typedef typename dax::cont::internal::Bindings<Invocation>::template GetType<N>::type ControlBinding;
  typedef typename dax::cont::arg::ConceptMapTraits<ControlBinding>::Tags Tags;
public:
  typedef typename boost::mpl::if_<
    typename Tags::template Has<dax::cont::sig::Point>,
    BindCellPoints<Invocation, N>,
    BindDirect<Invocation, N>
    >::type type;
};
} // namespace arg

}
}

#endif //__dax_exec_WorkletMapCell_h
