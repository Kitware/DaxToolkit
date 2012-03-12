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

#ifndef __dax_cuda_cont_ScheduleThrust_h
#define __dax_cuda_cont_ScheduleThrust_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/thrust/cont/ScheduleThrust.h>

namespace dax {
namespace cuda {
namespace cont {

template<class Functor, class Parameters>
DAX_CONT_EXPORT void scheduleThrust(Functor functor,
                                    Parameters parameters,
                                    dax::Id numInstances)
{
  dax::thrust::cont::scheduleThrust(functor, parameters, numInstances);
}

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_ScheduleThrust_h
