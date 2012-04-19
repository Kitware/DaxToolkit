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
#ifndef __dax_exec_math_LinearAlgebra_h
#define __dax_exec_math_LinearAlgebra_h

// This header file defines math functions that deal with linear albegra funcitons

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>
#include <dax/exec/math/Exp.h>

#ifndef DAX_CUDA
#include <math.h>
#endif

namespace dax {
namespace exec {
namespace math {

// ----------------------------------------------------------------------------
namespace internal {
template <typename T>
DAX_EXEC_EXPORT T normalize_template( T x ) {
  return x * dax::exec::math::RSqrt(dax::dot(x,x));
}
}

DAX_EXEC_EXPORT dax::Vector2 Normalize(dax::Vector2 x) {
  return internal::normalize_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Normalize(dax::Vector3 x) {
  return internal::normalize_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Normalize(dax::Vector4 x) {
  return internal::normalize_template(x);
}


}
}
} // namespace dax::exec::math

#undef DAX_SYS_MATH_FUNCTION
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_TEMPLATE

#endif //__dax_exec_math_LinearAlgebra_h
