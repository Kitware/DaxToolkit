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

#ifndef __dax_cont_SortDebug_h
#define __dax_cont_SortDebug_h

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>
#include <algorithm>

namespace dax {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void sortDebug(
    dax::cont::internal::ArrayContainerExecutionCPU<T> &values)
{
  std::sort(values.begin(),values.end());
}

}
} // namespace dax::cont

#endif //__dax_cont_SortDebug_h
