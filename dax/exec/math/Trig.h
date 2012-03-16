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
#ifndef __dax_exec_math_Trig_h
#define __dax_exec_math_Trig_h

// This header file defines math functions that deal with trigonometry.

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>

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
      return dax::exec::VectorMap(x, func ## _functor()); \
    } \
  }

//-----------------------------------------------------------------------------
/// Returns the constant Pi.
///
DAX_EXEC_EXPORT dax::Scalar Pi()
{
  return dax::Scalar(3.14159265358979323846264338327950288);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(sin)
DAX_SYS_MATH_TEMPLATE(cos)
DAX_SYS_MATH_TEMPLATE(tan)

/// Compute the sine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Sin(dax::Scalar x) {
  return internal::sin_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Sin(dax::Vector3 x) {
  return internal::sin_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Sin(dax::Vector4 x) {
  return internal::sin_template(x);
}

/// Compute the cosine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Cos(dax::Scalar x) {
  return internal::cos_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Cos(dax::Vector3 x) {
  return internal::cos_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Cos(dax::Vector4 x) {
  return internal::cos_template(x);
}

/// Compute the tangent of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Tan(dax::Scalar x) {
  return internal::tan_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Tan(dax::Vector3 x) {
  return internal::tan_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Tan(dax::Vector4 x) {
  return internal::tan_template(x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(asin)
DAX_SYS_MATH_TEMPLATE(acos)
DAX_SYS_MATH_TEMPLATE(atan)

/// Compute the arc sine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ASin(dax::Scalar x) {
  return internal::asin_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ASin(dax::Vector3 x) {
  return internal::asin_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ASin(dax::Vector4 x) {
  return internal::asin_template(x);
}

/// Compute the arc cosine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ACos(dax::Scalar x) {
  return internal::acos_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ACos(dax::Vector3 x) {
  return internal::acos_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ACos(dax::Vector4 x) {
  return internal::acos_template(x);
}

/// Compute the arc tangent of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ATan(dax::Scalar x) {
  return internal::atan_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ATan(dax::Vector3 x) {
  return internal::atan_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ATan(dax::Vector4 x) {
  return internal::atan_template(x);
}

//-----------------------------------------------------------------------------
/// Compute the arc tangent of \p y / \p x using the signs of both arguments
/// to determine the quadrant of the return value.
///
DAX_EXEC_EXPORT dax::Scalar ATan2(dax::Scalar y, dax::Scalar x)
{
  return DAX_SYS_MATH_FUNCTION(atan2)(y, x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(sinh)
DAX_SYS_MATH_TEMPLATE(cosh)
DAX_SYS_MATH_TEMPLATE(tanh)

/// Compute the hyperbolic sine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar SinH(dax::Scalar x) {
  return internal::sinh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 SinH(dax::Vector3 x) {
  return internal::sinh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 SinH(dax::Vector4 x) {
  return internal::sinh_template(x);
}

/// Compute the hyperbolic cosine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar CosH(dax::Scalar x) {
  return internal::cosh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 CosH(dax::Vector3 x) {
  return internal::cosh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 CosH(dax::Vector4 x) {
  return internal::cosh_template(x);
}

/// Compute the hyperbolic tangent of \p x.
///
DAX_EXEC_EXPORT dax::Scalar TanH(dax::Scalar x) {
  return internal::tanh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 TanH(dax::Vector3 x) {
  return internal::tanh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 TanH(dax::Vector4 x) {
  return internal::tanh_template(x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(asinh)
DAX_SYS_MATH_TEMPLATE(acosh)
DAX_SYS_MATH_TEMPLATE(atanh)

/// Compute the hyperbolic arc sine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ASinH(dax::Scalar x) {
  return internal::asinh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ASinH(dax::Vector3 x) {
  return internal::asinh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ASinH(dax::Vector4 x) {
  return internal::asinh_template(x);
}

/// Compute the hyperbolic arc cosine of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ACosH(dax::Scalar x) {
  return internal::acosh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ACosH(dax::Vector3 x) {
  return internal::acosh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ACosH(dax::Vector4 x) {
  return internal::acosh_template(x);
}

/// Compute the hyperbolic arc tangent of \p x.
///
DAX_EXEC_EXPORT dax::Scalar ATanH(dax::Scalar x) {
  return internal::atanh_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ATanH(dax::Vector3 x) {
  return internal::atanh_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ATanH(dax::Vector4 x) {
  return internal::atanh_template(x);
}

}
}
} // namespace dax::exec::math

#undef DAX_SYS_MATH_FUNCTION
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_TEMPLATE

#endif //__dax_exec_math_Trig_h
