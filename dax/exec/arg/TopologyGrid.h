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
#ifndef __dax_exec_arg_TopologyGrid_h
#define __dax_exec_arg_TopologyGrid_h

#include <dax/Types.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/Assert.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {

template <typename Tags, typename TopologyExec, typename TopologyConstExec>
struct TopologyGrid
{
protected:
  //What we have to do is use mpl::if_ to determine the type for
  //ExecArg
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   TopologyExec,
                                   TopologyConstExec>::type TopologyType;

  //if we are going with Out tag we create a value storage that holds a copy
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   typename TopologyExec::CellType,
                                   typename TopologyConstExec::CellType const>::type ReferenceType;
public:
  TopologyType Topo;

  typedef ReferenceType ReturnType;

  TopologyGrid(): Topo(){}

  template< typename Worklet>
  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index, const Worklet& work)
    {
    //if we have the In tag we have local store so use that value,
    //otherwise call the portal directly
    (void)work;  // Shut up compiler.
    DAX_ASSERT_EXEC(index >= 0, work);
    DAX_ASSERT_EXEC(index < this->Topo.GetNumberOfValues(), work);
    return ReturnType(this->Topo, index);
    }

  template< typename Worklet>
  DAX_EXEC_EXPORT void SaveExecutionResult(int, const Worklet&) const
    {

    }

  template< typename Worklet>
  DAX_EXEC_EXPORT void SaveExecutionResult(int, ReferenceType, const Worklet&) const
    {
    }
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_TopologyGrid_h
