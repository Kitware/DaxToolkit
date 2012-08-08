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
#ifndef __dax_cont_internal_ArrayContainerControlError_h
#define __dax_cont_internal_ArrayContainerControlError_h


namespace dax {
namespace cont {
namespace internal {

/// This is an invalid ArrayContainerControl. The point of this class is to
/// include the header file to make this invalid class the default
/// ArrayContainerControl. From that point, you have to specify an appropriate
/// ArrayContainerControl or else get a compile error.
///
struct ArrayContainerControlTagError
{
  // Not implemented.
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayContainerControlError_h
