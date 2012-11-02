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
#ifndef __dax_exec_WorkletGenerateTopology_h
#define __dax_exec_WorkletGenerateTopology_h

#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/exec/arg/Bind.h>
#include <dax/exec/arg/BindCellPoints.h>
#include <dax/exec/arg/FieldPortal.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace dax {
namespace exec {

///----------------------------------------------------------------------------
/// Superclass for worklets that generate new cell based topology. Use this when the worklet
/// needs to create new cells topology, with access to cell and point based information
///
class WorkletGenerateTopology : public dax::exec::internal::WorkletBase
{
public:
  typedef WorkletGenerateTopology WorkType;
  typedef dax::cont::sig::Cell DomainType;

  DAX_EXEC_EXPORT WorkletGenerateTopology() { }
protected:
  typedef dax::cont::arg::Field Field;
  typedef dax::cont::arg::Topology Topology;
  typedef dax::cont::sig::Cell Cell;
  typedef dax::cont::sig::Point Point;
  typedef dax::cont::sig::VisitIndex VisitIndex;
};

namespace arg {

template <int N, typename Invocation>
class Bind<WorkletGenerateTopology, dax::cont::sig::Arg<N>, Invocation>
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

#endif //__dax_exec_WorkletGenerateTopology_h
