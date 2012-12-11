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
#ifndef __dax_cont_scheduling_CreateExecutionResources_h
#define __dax_cont_scheduling_CreateExecutionResources_h

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>

namespace dax { namespace cont { namespace scheduling {

class CreateExecutionResources
{
protected:
  const dax::Id NumElementsToAlloc;
public:
  CreateExecutionResources(dax::Id size):
    NumElementsToAlloc(size)
    {}

  template <typename C, typename A>
  void operator()(dax::cont::arg::ConceptMap<C,A>& concept) const
    {
    //we must call to execution on everything.
    concept.ToExecution(NumElementsToAlloc);
    }
};

} } } //dax::cont::scheduling

#endif //__dax_cont_scheduling_CreateExecutionResources_h
