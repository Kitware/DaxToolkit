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
#ifndef __dax_internal_Invocation_h
#define __dax_internal_Invocation_h

#include <dax/internal/ParameterPack.h>

namespace dax {
namespace internal {

/// \brief Container for schedule invocation of worklets
///
/// When a dispatcher invokes a worklet, it has to create several templated
/// classes that depend on how the invoke method was called. In particular,
/// they need to know the type of the worklet and the type of the parameters
/// passed to Invoke on the control side. This simple struct wraps up those
/// pieces of information into a single template parameter.
///
template<typename _Worklet, typename _ControlInvocationParameters>
struct Invocation
{
  typedef _Worklet Worklet;
  typedef _ControlInvocationParameters ControlInvocationParameters;

  typedef typename dax::internal::ParameterPackToSignature<
    ControlInvocationParameters>::type ControlInvocationSignature;

  static const int NUM_PARAMETERS = ControlInvocationParameters::NUM_PARAMETERS;
};

}
} // namespace dax::internal

#endif //__dax_internal_Invocation_h
