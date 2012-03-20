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

#ifndef __dax_openmp_cont_InclusiveScan_h
#define __dax_openmp_cont_InclusiveScan_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>
#include <dax/thrust/cont/InclusiveScan.h>

namespace dax {
namespace openmp {
namespace cont {


template<typename T>
DAX_CONT_EXPORT typename T::ValueType inclusiveScan(const T &from, T& to)
{
  return dax::thrust::cont::inclusiveScan(from,to);
}

}
}
} // namespace dax::openmp::cont

#endif //__dax_openmp_cont_InclusiveScan_h
