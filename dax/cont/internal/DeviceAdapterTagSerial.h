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
#ifndef __dax_cont_internal_DeviceAdapterTagSerial_h
#define __dax_cont_internal_DeviceAdapterTagSerial_h

namespace dax {
namespace cont {

// This is intentionally in the dax::cont namespace instead of the
// dax::cont::internal namespace. Conceptually, this tag is defined in
// DeviceAdapterSerial.h, but is broken into this header to resolve some
// dependency issues.

/// A simple implementation of a DeviceAdapter that can be used for debuging.
/// The scheduling will simply run everything in a serial loop, which is easy
/// to track in a debugger.
///
struct DeviceAdapterTagSerial {  };

}
} // namespace dax::cont

#endif //__dax_cont_internal_DeviceAdapterTagSerial_h
