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
/// \brief Map an unstructured grid to an execution side cell topology parameter
template <typename Tags,
          typename Cell,
          typename CellContainerTag,
          typename PointContainerTag,
          typename DeviceTag
          >
class ConceptMap<Topology(Tags), dax::cont::UnstructuredGrid< Cell,
                                 CellContainerTag,PointContainerTag,
                                 DeviceTag > >
{
  typedef dax::cont::UnstructuredGrid< Cell,
          CellContainerTag, PointContainerTag, DeviceTag > GridType;

  //use mpl::if_ to determine the type for ExecArg
  typedef typename boost::mpl::if_<
      typename Tags::template Has<dax::cont::sig::Out>,
      typename GridType::TopologyStructExecution,
      typename GridType::TopologyStructConstExecution>::type TopologyType;

  typedef dax::exec::arg::TopologyCell<Tags,TopologyType> ExecGridType;
  GridType Grid;
  TopologyType Topology;

public:
  //All Topology binding classes must export the cell tag and grid tag
  //This allows us to do better scheduling based on cell / grid types
  typedef typename GridType::CellTag CellTypeTag;
  typedef typename GridType::GridTypeTag GridTypeTag;

  typedef GridType ContArg;
  typedef ExecGridType ExecArg;
  typedef dax::cont::sig::Cell DomainTag;

  ConceptMap(GridType g): Grid(g) {}

  ExecArg GetExecArg() const { return ExecGridType(this->Topology); }

  //All topology fields are required by dispatchers to expose the cont arg
  DAX_CONT_EXPORT const ContArg& GetContArg() const { return this->Grid; }

  void ToExecution(dax::Id size, boost::true_type)
    { /* Output */
    this->Topology = this->Grid.PrepareForOutput(size);
    }

  void ToExecution(dax::Id, boost::false_type)
    { /* Input  */
    this->Topology = this->Grid.PrepareForInput();
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

/// \headerfile TopologyUnstructured.h dax/cont/arg/TopologyUnstructuredGrid.h
/// \brief Map an unstructured grid to an execution side cell topology parameter
template <typename Tags,
          typename Cell,
          typename CellContainerTag,
          typename PointContainerTag,
          typename DeviceTag
          >
class ConceptMap<Topology(Tags), const dax::cont::UnstructuredGrid< Cell,
                                 CellContainerTag,PointContainerTag,
                                 DeviceTag > >
{
  typedef dax::cont::UnstructuredGrid< Cell,
          CellContainerTag, PointContainerTag, DeviceTag > GridType;

  //use mpl::if_ to determine the type for ExecArg
  typedef typename GridType::TopologyStructConstExecution TopologyType;

  typedef dax::exec::arg::TopologyCell<Tags,TopologyType> ExecGridType;
  GridType Grid;
  TopologyType Topology;

public:
  //All Topology binding classes must export the cell tag and grid tag
  //This allows us to do better scheduling based on cell / grid types
  typedef typename GridType::CellTag CellTypeTag;
  typedef typename GridType::GridTypeTag GridTypeTag;

  typedef GridType ContArg;
  typedef ExecGridType ExecArg;
  typedef dax::cont::sig::Cell DomainTag;

  ConceptMap(GridType g): Grid(g) {}

  ExecArg GetExecArg() const { return ExecGridType(this->Topology); }

  //All topology fields are required by dispatchers to expose the cont arg
  DAX_CONT_EXPORT const ContArg& GetContArg() const { return this->Grid; }

  void ToExecution(dax::Id, boost::false_type)
    { /* Input  */
    this->Topology = this->Grid.PrepareForInput();
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
