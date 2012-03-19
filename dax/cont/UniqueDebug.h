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


#ifndef __dax_cont_UniqueDebug_h
#define __dax_cont_UniqueDebug_h

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>

#include <algorithm>

namespace dax {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void uniqueDebug(dax::cont::internal::ArrayContainerExecutionCPU<T> &values)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::iterator resultType;

  resultType newEnd = std::unique(values.begin(),values.end());
  values.Allocate( std::distance(values.begin(),newEnd) );

}

}
} // namespace dax::cont

#endif //__dax_cont_UniqueDebug_h
