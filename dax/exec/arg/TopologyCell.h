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
#ifndef __dax_exec_arg_TopologyCell_h
#define __dax_exec_arg_TopologyCell_h

#include <dax/Types.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/Assert.h>

#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyType>
class TopologyCell
{
  TopologyType Topo;
  typename TopologyType::CellType Cell;
public:
  typedef typename TopologyType::CellType CellType;
  typedef typename CellType::PointConnectionsType SaveType;

  //if we are going with Out tag we create a value storage that holds a copy
  //otherwise we have to pass a copy, since portals don't have to provide a reference
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   CellType&,
                                   CellType const&>::type ReturnType;

  TopologyCell(const TopologyType& t): Topo(t), Cell(t){}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index,
                            const dax::exec::internal::WorkletBase& work)
    {
    //if we have the In tag we have local store so use that value,
    //otherwise call the portal directly
    (void)work;  // Shut up compiler.
    DAX_ASSERT_EXEC(index >= 0, work);
    this->Cell.SetPointIndices(this->Topo,index);
    return this->Cell;
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int,
                       const dax::exec::internal::WorkletBase&) const
    {
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int index, const SaveType& values,
                       const dax::exec::internal::WorkletBase& work) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our TopoExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
        template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(index,values,work,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id index,
                  const SaveType& values,
                  dax::exec::internal::WorkletBase work,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::exec::internal::FieldSetMultiple(this->Topo.CellConnections,
                                        CellType::NUM_POINTS * index,
                                        values,
                                        work);
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id,
                const SaveType&,
                dax::exec::internal::WorkletBase,
                HasOutTag,
                typename boost::disable_if<HasOutTag>::type* = 0) const
    {
    }
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_TopologyCell_h
