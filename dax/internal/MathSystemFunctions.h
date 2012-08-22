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
#ifndef __dax_internal_MathSystemFunctions_h
#define __dax_internal_MathSystemFunctions_h

// This header file contains helper macros and templates for wrapping math
// functions in a portable way.

#include <dax/Types.h>
#include <dax/VectorOperations.h>

#ifndef DAX_CUDA
#include <math.h>
#endif

/// The DAX_SYS_MATH_FUNCTION macro mangles a math function name so that it is
/// the correct one to apply to the dax::Scalar type. (Specifically, it adds an
/// f suffix when dax::Scalar is a float type, nothing when it is double.)
///
#ifdef DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func
#else //DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func ## f
#endif //DAX_USE_DOUBLE_PRECISION

namespace dax {
namespace internal {

// For those of you not familar with some of the possibly crazy-ass C++ syntax,
// below is a template parameter defined as a function pointer. We could make
// this a little bit easier by simply making a standard class argument and
// passing the pointer, but that could make it harder for the compiler to
// inline the call.

/// A simple struct that turns a system math-like function into a functor
/// object.  Used internally in SysMathVectorCall
///
template<dax::Scalar (*SysMathFunc)(dax::Scalar)>
struct SysMathFunctor {
  DAX_EXEC_CONT_EXPORT dax::Scalar operator()(dax::Scalar x) const {
    return SysMathFunc(x);
  }
};

/// Applies a system math-like function to a vector-like type.
///
template<dax::Scalar (*SysMathFunc)(dax::Scalar), typename T>
DAX_EXEC_CONT_EXPORT T SysMathVectorCall(T x)
{
  return dax::VectorMap(x, dax::internal::SysMathFunctor<SysMathFunc>());
}

}
} // namespace dax::internal

#endif //__dax_internal_MathSystemFunctions_h
