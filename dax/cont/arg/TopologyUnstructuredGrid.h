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
#ifndef __dax_cont_arg_TopologyUnstructuredGrid_h
#define __dax_cont_arg_TopologyUnstructuredGrid_h

#include <dax/Types.h>
#include <dax/internal/Tags.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/arg/TopologyCell.h>
#include <dax/cont/UnstructuredGrid.h>

namespace dax { namespace cont { namespace arg {

/// \headerfile TopologyUnstructured.h dax/cont/arg/TopologyUnstructuredGrid.h
/// \brief Map an uniform grid to an execution side cell topology parameter
template <typename Tags, typename Cell, typename ContainerTag, typename DeviceTag >
class ConceptMap<Topology(Tags), dax::cont::UnstructuredGrid< Cell,ContainerTag,DeviceTag > >
{
private:
  typedef dax::cont::UnstructuredGrid< Cell,ContainerTag,DeviceTag > GridType;
  typedef dax::exec::arg::TopologyCell<Tags,
            typename GridType::TopologyStructExecution,
            typename GridType::TopologyStructConstExecution > ExecGridType;
  GridType Grid;
  ExecGridType ExecArg_;

public:
  typedef ExecGridType ExecArg;
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Cell>::Tags DomainTags;

  ConceptMap(GridType g):
    Grid(g),
    ExecArg_()
    {}

  ExecArg& GetExecArg() { return this->ExecArg_; }

  void ToExecution(dax::Id size, boost::true_type)
    { /* Output */
    this->ExecArg_.Topo = this->Grid.PrepareForOutput(size);
    }

  void ToExecution(dax::Id, boost::false_type)
    { /* Input  */
    this->ExecArg_.Topo = this->Grid.PrepareForInput();
    }

  //we need to pass the number of elements to allocate
  void ToExecution(dax::Id size)
    {
    ToExecution(size,typename Tags::template Has<dax::cont::sig::Out>());
    }

  dax::Id GetDomainLength(sig::Point) const
    {
    return Grid.GetNumberOfPoints();
    }

  dax::Id GetDomainLength(sig::Cell) const
    {
    return Grid.GetNumberOfCells();
    }
};

}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_TopologyUnstructuredGrid_h
