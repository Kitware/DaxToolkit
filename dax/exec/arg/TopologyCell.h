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
#include <dax/exec/CellVertices.h>

#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {


//TopologyCell is a base class type that is never actually used. Instead
//It is always wrapped by a BindCell* class. Be it BindCellTag or BindCellVertices
template <typename Tags, typename TopologyType>
class TopologyCell
{
public:
  typedef typename TopologyType::CellTag CellTag;
  typedef dax::exec::CellVertices<CellTag> CellVerticesType;

  //if we are going with Out tag
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   CellVerticesType&,
                                   CellVerticesType const&>::type ReturnType;

  typedef CellVerticesType SaveType;
  typedef CellVerticesType ValueType;

  DAX_CONT_EXPORT TopologyCell(const TopologyType& t): Topo(t), Cell(0) {  }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(
      const IndexType& index,
      const dax::exec::internal::WorkletBase& work)
  {
    (void)work;  // Shut up compiler.
    DAX_ASSERT_EXEC(index >= 0, work);
    DAX_ASSERT_EXEC(index < Topo.GetNumberOfCells(), work);
    this->Cell = this->Topo.GetCellConnections(index);
    return this->Cell;
  }

  DAX_EXEC_EXPORT void SaveExecutionResult(int index,
                       const dax::exec::internal::WorkletBase& work) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our TopoExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
        template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(index,this->Cell,work,HasOutTag());
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int index, const SaveType& v,
                       const dax::exec::internal::WorkletBase& work) const
    {
    //Look at the concept map traits. If we have the Out tag
    //we know that we must call our TopoExecArgs SaveExecutionResult.
    //Otherwise we are an input argument and that behavior is undefined
    //and very bad things could happen
    typedef typename Tags::
        template Has<typename dax::cont::sig::Out>::type HasOutTag;
    this->saveResult(index,v,work,HasOutTag());
    }

  //method enabled when we do have the out tag ( or InOut)
  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id index,
                  const SaveType &values,
                  dax::exec::internal::WorkletBase work,
                  HasOutTag,
                  typename boost::enable_if<HasOutTag>::type* = 0) const
    {
    dax::exec::internal::FieldSetMultiple(this->Topo.CellConnections,
                                        dax::CellTraits<CellTag>::NUM_VERTICES * index,
                                        values.GetAsTuple(),
                                        work);
    }

  template <typename HasOutTag>
  DAX_EXEC_EXPORT
  void saveResult(dax::Id,
                  const SaveType &,
                  dax::exec::internal::WorkletBase,
                  HasOutTag,
                  typename boost::disable_if<HasOutTag>::type* = 0) const
    {
    }

  TopologyType Topo;
private:
  CellVerticesType Cell;
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_TopologyCell_h
