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
#ifndef __dax_thrust_cont_internal_DeviceAdapterThrustTag_h
#define __dax_thrust_cont_internal_DeviceAdapterThrustTag_h

namespace dax {
namespace thrust {
namespace cont {
namespace internal {

/// This is an incomplete implementation of a device adapter. Specifically, it
/// is missing the ArrayManagerExecution and ExecutionAdapter classes and the
/// schedule function. This is why it is declared in an internal namespace.
/// However, all device adapters based on thrust should have their tags inherit
/// from this tag. This will make the implementation of the device adapter
/// algorithms automatically use those defined here.
///
/// The thrust namespace contains basic implementations of all missing parts.
/// The ArrayManagerExecution should be a trivial subclass of either
/// ArrayManagerExecutionThrustDevice or ArrayManagerExecutionThrustShare.
/// There is a basic type for ExecutionAdapter called ExecutionAdapterThrust
/// that thrust implementations should publicly subclass. Likewise, there is a
/// function called ScheduleThrust that implementations should trivially call
/// for their schedule implementation.
///
struct DeviceAdapterTagThrust {  };

}
}
}
} // namespace dax::thrust::cont::internal

#endif //__dax_thrust_cont_internal_DeviceAdapterThrustTag_h
