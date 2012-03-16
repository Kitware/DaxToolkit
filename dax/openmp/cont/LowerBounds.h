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

#ifndef __dax_openmp_cont_LowerBounds_h
#define __dax_openmp_cont_LowerBounds_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>
#include <dax/thrust/cont/LowerBounds.h>

namespace dax {
namespace openmp {
namespace cont {


template<typename T, typename U>
DAX_CONT_EXPORT void lowerBounds(const T& input,
                                 const T& values,
                                 U& output)
{
  dax::thrust::cont::lowerBounds(input,values,output);
}

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_LowerBounds_h
