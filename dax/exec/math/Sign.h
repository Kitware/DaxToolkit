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
#ifndef __dax_exec_math_Sign_h
#define __dax_exec_math_Sign_h

// This header file defines math functions that deal with the sign (positive or
// negative) of numbers.

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>

#ifndef DAX_CUDA
#include <math.h>
#include <stdlib.h>
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
      return dax::exec::VectorMap(x, func ## _functor()); \
    } \
  }

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(fabs)

/// Return the absolute value of \x. That is, return \p x if it is positive or
/// \p -x if it is negative.
///
DAX_EXEC_EXPORT dax::Scalar Abs(dax::Scalar x) {
  return internal::fabs_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Abs(dax::Vector3 x) {
  return internal::fabs_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Abs(dax::Vector4 x) {
  return internal::fabs_template(x);
}
DAX_EXEC_EXPORT dax::Id Abs(dax::Id x)
{
#if DAX_SIZE_ID == DAX_SIZE_INT
  return abs(x);
#elif DAX_SIZE_ID == DAX_SIZE_LONG
  return labs(x);
#elif DAX_SIZE_ID == DAX_SIZE_LONG_LONG
  return llabs(x);
#else
#error Cannot find correct size for dax::Id.
#endif
}
DAX_EXEC_EXPORT dax::Id3 Abs(dax::Id3 x)
{
  return dax::make_Id3(Abs(x[0]), Abs(x[1]), Abs(x[2]));
}

//-----------------------------------------------------------------------------
/// Returns true if \p x is less than zero, false otherwise.
///
DAX_EXEC_EXPORT bool IsNegative(dax::Scalar x)
{
  return (signbit(x) != 0);
}

//-----------------------------------------------------------------------------
/// Copies the sign of \p y onto \p x.  If \p y is positive, returns Abs(\p x).
/// If \p x is negative, returns -Abs(\p x).
///
DAX_EXEC_EXPORT dax::Scalar CopySign(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(copysign)(x, y);
}

}
}
} // namespace dax::exec::math

#undef DAX_SYS_MATH_FUNCTION
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_TEMPLATE

#endif //__dax_exec_math_Sign_h
