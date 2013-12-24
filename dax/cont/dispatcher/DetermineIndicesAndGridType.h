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
#ifndef __dax_cont_dispatcher_DetermineIndicesAndGridType_h
#define __dax_cont_dispatcher_DetermineIndicesAndGridType_h

#include <dax/Extent.h>
#include <dax/cont/arg/Topology.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/FindBinding.h>
#include <dax/cont/internal/GridTags.h>

#include <dax/exec/WorkletMapCell.h>

namespace dax { namespace cont { namespace dispatcher {

namespace internal
{
  template< class GridTagType >
  struct DetermineGridIndexType
  {
    typedef dax::Id type;
  };

  template<>
  struct DetermineGridIndexType< dax::cont::internal::UniformGridTag >
  {
    typedef dax::Id3 type;
  };

  template< class GridTypeTag>
  struct GenerateGridCount
  {
    typedef typename DetermineGridIndexType<GridTypeTag>::type ReturnType;

    template<class Topo>
    ReturnType operator()(const Topo& t) const { return t.GetNumberOfCells(); }
  };

  template<>
  struct GenerateGridCount< dax::cont::internal::UniformGridTag >
  {
    typedef dax::cont::internal::UniformGridTag GridTypeTag;
    typedef DetermineGridIndexType<GridTypeTag>::type ReturnType;

    template<class Topo>
    ReturnType operator()(const Topo& t) const
      {
      return dax::extentCellDimensions(t.GetExtent());
      }
  };
}

//the default is that the worklet isn't a candidate for grid scheduling
template<typename WorkletBaseType, typename Invocation>
class DetermineIndicesAndGridType
{
  typedef typename dax::cont::internal::Bindings<Invocation>::type BindingsType;
  const dax::Id NumInstances;

public:
  DetermineIndicesAndGridType(const BindingsType& daxNotUsed(bindings),
                              dax::Id numInstances ):
    NumInstances(numInstances)
    {
    }

  //return the proper exec object that can be used to dispatch
  dax::Id gridCount() const
  {
    return NumInstances;
  }

 bool isValidForGridScheduling() const
    { return false; }
};


//worklet map cell is a candidate for grid scheduling.
template<typename Invocation>
class DetermineIndicesAndGridType<dax::exec::WorkletMapCell,
                                  Invocation>
{
  // Determine the topology type by finding the topo binding. First
  // we look up the index of the binding, the second step is to actually
  // get the type for that binding argument.
  typedef typename dax::cont::internal::FindBinding<
                                  Invocation,
                                  dax::cont::arg::Topology>::type TopoIndex;
  typedef typename dax::cont::internal::Bindings<Invocation>::type BindingsType;
  typedef typename BindingsType::template GetType<
                                  TopoIndex::value>::type TopoControlBinding;

  //2: now that we have the execution arg object we can extract the grid
  //   when it is passed into the constructor
  typedef typename TopoControlBinding::ContArg TopoContArgType;

  const TopoContArgType& Topology;
  const dax::Id NumInstances;

public:
  //expose the grid and cell tag types and the grid index type
  typedef typename TopoControlBinding::CellTypeTag  CellTypeTag;
  typedef typename TopoControlBinding::GridTypeTag  GridTypeTag;
  typedef typename internal::DetermineGridIndexType< GridTypeTag >::type
                                                    GridIndexType;

  DetermineIndicesAndGridType(const BindingsType& bindings,
                              dax::Id numInstances):
    Topology(bindings.template Get<TopoIndex::value>().GetContArg()),
    NumInstances(numInstances)
    {
    }

  //return the proper exec object that can be used to dispatch
  GridIndexType gridCount() const
  {
    return internal::GenerateGridCount<GridTypeTag>()(this->Topology);
  }

  bool isValidForGridScheduling() const
    {
    //this is really important to make sure that we are iterating is equal
    //to the number of cells, because if it isn't that means we have some
    //permutation of the cells which will break the optimized cell schedulers
    return this->NumInstances == this->Topology.GetNumberOfCells();
    }

};

} } } //namespace dax::cont::dispatcher
#endif
