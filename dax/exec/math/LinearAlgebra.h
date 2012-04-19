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

#ifdef DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func
#else //DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func ## f
#endif //DAX_USE_DOUBLE_PRECISION

#define DAX_SYS_MATH_FUNCTOR(func) \
  namespace internal { \
  struct func ## _functor { \
    DAX_EXEC_EXPORT dax::Scalar operator()(dax::Scalar x) const { \
      return DAX_SYS_MATH_FUNCTION(func)(x); \
    } \
  }; \
  }

#define DAX_SYS_MATH_TEMPLATE(func) \
  DAX_SYS_MATH_FUNCTOR(func) \
  namespace internal { \
    template <typename T> DAX_EXEC_EXPORT T func ## _template(T x) \
    { \
      return func(x); \
    } \
  }

// ----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_TEMPLATE( normalize )
#else
namespace internal {
struct add_functor {
  DAX_EXEC_EXPORT dax::Scalar  operator () (const dax::Scalar &x ,
                                            const dax::Scalar &y) const {
    return x+y;
  }
};
template <typename T>
DAX_EXEC_EXPORT dax::Scalar normalize_template( T x ) {
  return dax::exec::math::Sqrt(dax::exec::VectorReduce(x*x, add_functor() ) );
}
}
#endif

DAX_EXEC_EXPORT dax::Scalar Normalize(dax::Scalar x) {
  return internal::normalize_template(x);
}
DAX_EXEC_EXPORT dax::Scalar Normalize(dax::Vector2 x) {
  return internal::normalize_template(x);
}
DAX_EXEC_EXPORT dax::Scalar Normalize(dax::Vector3 x) {
  return internal::normalize_template(x);
}
DAX_EXEC_EXPORT dax::Scalar Normalize(dax::Vector4 x) {
  return internal::normalize_template(x);
}


}
}
} // namespace dax::exec::math

#undef DAX_SYS_MATH_FUNCTION
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_TEMPLATE

#endif //__dax_exec_math_LinearAlgebra_h
