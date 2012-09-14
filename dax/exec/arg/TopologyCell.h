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

#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyExec, typename TopologyConstExec>
class TopologyCell
{

  //What we have to do is use mpl::if_ to determine the type for
  //ExecArg
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   TopologyExec,
                                   TopologyConstExec>::type TopologyType;

  //if we are going with Out tag we create a value storage that holds a copy
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   typename TopologyExec::CellType,
                                   typename TopologyConstExec::CellType>::type ReferenceType;

public:
  TopologyType Topo;

  typedef ReferenceType ReturnType;
  typedef ReferenceType CellType;

  TopologyCell(): Topo(){}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index,
                            const dax::exec::internal::WorkletBase& work)
    {
    //if we have the In tag we have local store so use that value,
    //otherwise call the portal directly
    (void)work;  // Shut up compiler.
    DAX_ASSERT_EXEC(index >= 0, work);
    return CellType(this->Topo,index);
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int,
                       const dax::exec::internal::WorkletBase&) const
    {

    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int, ReturnType,
                       const dax::exec::internal::WorkletBase&) const
    {
    }
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_TopologyCell_h
