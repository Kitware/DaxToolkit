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

#ifndef __dax_cont_LowerBoundsSerial_h
#define __dax_cont_LowerBoundsSerial_h

#include <dax/Types.h>
#include <dax/cont/internal/ArrayContainerExecutionCPU.h>

#include <algorithm>

namespace dax {
namespace cont {
template<typename T>
DAX_CONT_EXPORT void lowerBoundsSerial(
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& input,
    const dax::cont::internal::ArrayContainerExecutionCPU<T>& values,
    dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>& output)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::const_iterator ConstInputIter;
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::const_iterator InputIter;

  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>::iterator OutIter;

  //stl lower_bound only supports a single value to search for.
  //So we iterate over all the values and search for each one
  OutIter out=output.begin();
  ConstInputIter inputStartPos=input.begin(); //needed for distance call
  ConstInputIter result;
  for(ConstInputIter i=values.begin(); i!=values.end(); ++i,++out)
    {
    //std::lower_bound returns an iterator to the position where you can insert
    //we want the distance from the start
    result = std::lower_bound(input.begin(),input.end(),*i);
    *out = static_cast<dax::Id>(std::distance(inputStartPos,result));
    }
}


}
} // namespace dax::cont

#endif //__dax_cont_LowerBoundsSerial_h
