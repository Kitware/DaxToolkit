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

#if _WIN32 && !defined DAX_CUDA_COMPILATION
#include <boost/math/special_functions/acosh.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <boost/math/special_functions/atanh.hpp>
#define DAX_USE_BOOST_MATH

#endif

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
#ifdef DAX_USE_BOOST_MATH
  //windows doesn't have any of the c99 math functions
  return boost::math::asinh(x);
#else
  return DAX_SYS_MATH_FUNCTION(asinh)(x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ASinH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::ASinH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ASinH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::ASinH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ASinH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::ASinH>(x);
}

/// Compute the hyperbolic arc cosine of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ACosH(dax::Scalar x) {
#ifdef DAX_USE_BOOST_MATH
  //windows doesn't have any of the c99 math functions
  return boost::math::acosh(x);
#else
  return DAX_SYS_MATH_FUNCTION(acosh)(x);
#endif
}

DAX_EXEC_CONT_EXPORT dax::Vector2 ACosH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::ACosH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ACosH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::ACosH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ACosH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::ACosH>(x);
}

/// Compute the hyperbolic arc tangent of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ATanH(dax::Scalar x) {
#ifdef DAX_USE_BOOST_MATH
  //windows doesn't have any of the c99 math functions
  return boost::math::atanh(x);
#else
  return DAX_SYS_MATH_FUNCTION(atanh)(x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ATanH(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::ATanH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ATanH(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::ATanH>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ATanH(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::ATanH>(x);
}

}
} // namespace dax::math

#ifdef DAX_USE_BOOST_MATH
#undef DAX_USE_BOOST_MATH
#endif

#endif //__dax_math_Trig_h
