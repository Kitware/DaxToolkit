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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_ErrorControlInternal_h
#define __dax_cont_ErrorControlInternal_h

#include <dax/cont/ErrorControl.h>

namespace dax {
namespace cont {

/// This class is thrown when Dax detects an internal state that should never
/// be reached. This error usually indicates a bug in dax or, at best, Dax
/// failed to detect an invalid input it should have.
///
class ErrorControlInternal : public ErrorControl
{
public:
  ErrorControlInternal(const std::string &message)
    : ErrorControl(message) { }
};

}
} // namespace dax::cont

#endif //__dax_cont_ErrorControlInternal_h
