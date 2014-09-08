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
#ifndef __dax_cont_arg_GeometeryEdgeInterpolatedGrid_h
#define __dax_cont_arg_GeometeryEdgeInterpolatedGrid_h

#include <dax/Types.h>
#include <dax/CellTraits.h>
#include <dax/internal/Tags.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Geometry.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/arg/GeometryCell.h>
#include <dax/cont/internal/EdgeInterpolatedGrid.h>

namespace dax { namespace cont { namespace arg {

/// \headerfile GeometryUnstructured.h dax/cont/arg/GeometryEdgeInterpolatedGrid.h
/// \brief Map an unstructured grid to an execution side cell topology parameter
template <typename Tags,
          typename Cell,
          typename CellContainerTag,
          typename PointContainerTag,
          typename DeviceTag
          >
class ConceptMap<Geometry(Tags), dax::cont::internal::EdgeInterpolatedGrid< Cell,
                                 CellContainerTag,PointContainerTag,
                                 DeviceTag > >
{
  typedef dax::cont::internal::EdgeInterpolatedGrid< Cell,
          CellContainerTag, PointContainerTag, DeviceTag > GridType;

  //use mpl::if_ to determine the type for ExecArg
  typedef typename boost::mpl::if_<
    typename Tags::template Has<dax::cont::sig::Out>,
    typename GridType::TopologyStructExecution,
    typename GridType::TopologyStructConstExecution>::type TopologyType;

  typedef typename boost::mpl::if_<
    typename Tags::template Has<dax::cont::sig::Out>,
    typename GridType::InterpolatedPointsType::PortalExecution,
    typename GridType::InterpolatedPointsType::PortalConstExecution>::type EdgePortalType;

  typedef dax::exec::arg::GeometryCell<Tags,TopologyType,EdgePortalType> ExecGridType;

  GridType Grid;
  TopologyType Topology;
  EdgePortalType EdgeInterpolations;

public:
  //All Topology binding classes must export the cell tag and grid tag
  //This allows us to do better scheduling based on cell / grid types
  typedef typename GridType::CellTag CellTypeTag;
  typedef typename GridType::GridTypeTag GridTypeTag;

  typedef GridType ContArg;
  typedef ExecGridType ExecArg;
  typedef dax::cont::sig::Cell DomainTag;

  ConceptMap(GridType g): Grid(g) {}

  ExecArg GetExecArg() const {
    return ExecGridType(this->Topology,this->EdgeInterpolations);
  }

  //All topology fields are required by dispatcher to expose the cont arg
  DAX_CONT_EXPORT const ContArg& GetContArg() const { return this->Grid; }

  void ToExecution(dax::Id size, boost::true_type)
    { /* Output */
    this->Topology = this->Grid.PrepareForOutput(size);

    //find out the number of EdgeInterpolations per cell and allocate
    //to size * that.
    this->EdgeInterpolations = this->Grid.GetInterpolatedPoints().PrepareForOutput(
                     size *  dax::CellTraits<CellTypeTag>::NUM_VERTICES );
    }

  void ToExecution(dax::Id, boost::false_type)
    { /* Input  */
    this->Topology = this->Grid.PrepareForInput();
    this->EdgeInterpolations = this->Grid.GetInterpolatedPoints().PrepareForInput();
    }

  //we need to pass the number of elements to allocate
  void ToExecution(dax::Id size)
    {
    ToExecution(size,typename Tags::template Has<dax::cont::sig::Out>());
    }

  dax::Id GetDomainLength(sig::Point) const
    {
    return Grid.GetNumberOfEdgeInterpolations();
    }

  dax::Id GetDomainLength(sig::Cell) const
    {
    return Grid.GetNumberOfCells();
    }
};

/// \headerfile TopologyUnstructured.h dax/cont/arg/TopologyEdgeInterpolatedGrid.h
/// \brief Map an unstructured grid to an execution side cell topology parameter
template <typename Tags,
          typename Cell,
          typename CellContainerTag,
          typename PointContainerTag,
          typename DeviceTag
          >
class ConceptMap<Geometry(Tags), const dax::cont::internal::EdgeInterpolatedGrid< Cell,
                                 CellContainerTag,PointContainerTag,
                                 DeviceTag > >
{
  typedef dax::cont::internal::EdgeInterpolatedGrid< Cell,
          CellContainerTag, PointContainerTag, DeviceTag > GridType;

  //use mpl::if_ to determine the type for ExecArg
  typedef typename GridType::TopologyStructConstExecution TopologyType;

  typedef typename GridType::InterpolatedPointsType::PortalConstExecution EdgePortalType;

  typedef dax::exec::arg::GeometryCell<Tags,TopologyType,EdgePortalType> ExecGridType;
  GridType Grid;
  TopologyType Topology;
  EdgePortalType EdgeInterpolations;

public:
  //All Topology binding classes must export the cell tag and grid tag
  //This allows us to do better scheduling based on cell / grid types
  typedef typename GridType::CellTag CellTypeTag;
  typedef typename GridType::GridTypeTag GridTypeTag;

  typedef GridType ContArg;
  typedef ExecGridType ExecArg;
  typedef dax::cont::sig::Cell DomainTag;

  ConceptMap(GridType g): Grid(g) {}

  ExecArg GetExecArg() const {
    return ExecGridType(this->Topology,this->EdgeInterpolations);
  }

  //All topology fields are required by dispatcher to expose the cont arg
  DAX_CONT_EXPORT const ContArg& GetContArg() const { return this->Grid; }

  void ToExecution(dax::Id, boost::false_type)
    { /* Input  */
    this->Topology = this->Grid.PrepareForInput();
    this->EdgeInterpolations = this->Grid.GetInterpolatedPoints().PrepareForInput();
    }

  //we need to pass the number of elements to allocate
  void ToExecution(dax::Id size)
    {
    ToExecution(size,typename Tags::template Has<dax::cont::sig::Out>());
    }

  dax::Id GetDomainLength(sig::Point) const
    {
    return Grid.GetNumberOfEdgeInterpolations();
    }

  dax::Id GetDomainLength(sig::Cell) const
    {
    return Grid.GetNumberOfCells();
    }
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_TopologyEdgeInterpolatedGrid_h
