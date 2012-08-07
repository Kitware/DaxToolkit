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
#ifndef __dax_math_Trig_h
#define __dax_math_Trig_h

// This header file defines math functions that deal with trigonometry.

#include <dax/internal/MathSystemFunctions.h>

namespace dax {
namespace math {

//-----------------------------------------------------------------------------
/// Returns the constant Pi.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Pi()
{
  return dax::Scalar(3.14159265358979323846264338327950288);
}

//-----------------------------------------------------------------------------
/// Compute the sine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Sin(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Sin(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Sin(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Sin(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sin)>(x);
}

/// Compute the cosine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Cos(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Cos(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Cos(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Cos(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cos)>(x);
}

/// Compute the tangent of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Tan(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Tan(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Tan(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Tan(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tan)>(x);
}

//-----------------------------------------------------------------------------
/// Compute the arc sine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ASin(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ASin(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ASin(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asin)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ASin(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asin)>(x);
}

/// Compute the arc cosine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ACos(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ACos(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ACos(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acos)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ACos(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acos)>(x);
}

/// Compute the arc tangent of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ATan(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ATan(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ATan(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atan)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ATan(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atan)>(x);
}

//-----------------------------------------------------------------------------
/// Compute the arc tangent of \p y / \p x using the signs of both arguments
/// to determine the quadrant of the return value.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ATan2(dax::Scalar y, dax::Scalar x)
{
  return DAX_SYS_MATH_FUNCTION(atan2)(y, x);
}

//-----------------------------------------------------------------------------
/// Compute the hyperbolic sine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar SinH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 SinH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 SinH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 SinH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sinh)>(x);
}

/// Compute the hyperbolic cosine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar CosH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 CosH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 CosH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 CosH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cosh)>(x);
}

/// Compute the hyperbolic tangent of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar TanH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 TanH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 TanH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 TanH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(tanh)>(x);
}

//-----------------------------------------------------------------------------
/// Compute the hyperbolic arc sine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ASinH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ASinH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ASinH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asinh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ASinH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(asinh)>(x);
}

/// Compute the hyperbolic arc cosine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ACosH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ACosH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ACosH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acosh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ACosH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(acosh)>(x);
}

/// Compute the hyperbolic arc tangent of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ATanH(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ATanH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ATanH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atanh)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ATanH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(atanh)>(x);
}

}
} // namespace dax::math

#endif //__dax_math_Trig_h
