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
#ifndef __dax_math_Exp_h
#define __dax_math_Exp_h

// This header file defines math functions that deal with exponentials.

#include <dax/internal/MathSystemFunctions.h>

namespace dax {
namespace math {

//-----------------------------------------------------------------------------
/// Computes \p x raised to the power of \p y.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Pow(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(pow)(x, y);
}

//-----------------------------------------------------------------------------
/// Compute the square root of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Sqrt(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sqrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Sqrt(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sqrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Sqrt(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sqrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Sqrt(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(sqrt)>(x);
}

//-----------------------------------------------------------------------------
/// Compute the reciprocal square root of \p x. The result of this function is
/// equivalent to <tt>1/Sqrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal square root than the regular square root. Thus, you
/// should use this function whenever dividing by the square root.
///
DAX_EXEC_CONT_EXPORT dax::Scalar RSqrt(dax::Scalar x) {
#ifdef DAX_CUDA
  return DAX_SYS_MATH_FUNCTION(rsqrt)(x);
#else
  return 1/dax::math::Sqrt(x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 RSqrt(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::RSqrt>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 RSqrt(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::RSqrt>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 RSqrt(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::RSqrt>(x);
}

//-----------------------------------------------------------------------------
/// Compute the cube root of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Cbrt(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cbrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Cbrt(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cbrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Cbrt(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cbrt)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Cbrt(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(cbrt)>(x);
}

//-----------------------------------------------------------------------------
/// Compute the reciprocal cube root of \p x. The result of this function is
/// equivalent to <tt>1/Cbrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal cube root than the regular cube root. Thus, you
/// should use this function whenever dividing by the cube root.
///
DAX_EXEC_CONT_EXPORT dax::Scalar RCbrt(dax::Scalar x) {
#ifdef DAX_CUDA
  return DAX_SYS_MATH_FUNCTION(rcbrt)(x);
#else
  return 1/dax::math::Cbrt(x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 RCbrt(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::RCbrt>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 RCbrt(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::RCbrt>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 RCbrt(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::RCbrt>(x);
}

//-----------------------------------------------------------------------------
/// Computes e**\p x, the base-e exponential of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Exp(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Exp(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Exp(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Exp(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp)>(x);
}

/// Computes 2**\p x, the base-2 exponential of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Exp2(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Exp2(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Exp2(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Exp2(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(exp2)>(x);
}

/// Computes (e**\p x) - 1, the of base-e exponental of \p x then minus 1. The
/// accuracy of this function is good even for very small values of x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ExpM1(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(expm1)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ExpM1(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(expm1)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ExpM1(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(expm1)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ExpM1(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(expm1)>(x);
}

//-----------------------------------------------------------------------------
/// Computes 10**\p x, the base-10 exponential of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Exp10(dax::Scalar x) {
#ifdef DAX_CUDA
  return DAX_SYS_MATH_FUNCTION(exp10)(x);
#else
  return dax::math::Pow(10, x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Exp10(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::Exp10>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Exp10(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::Exp10>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Exp10(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::Exp10>(x);
}

//-----------------------------------------------------------------------------
/// Computes the natural logarithm of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Log(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Log(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Log(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Log(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log)>(x);
}

/// Computes the logarithm base 2 of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Log2(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Log2(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Log2(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log2)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Log2(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log2)>(x);
}

/// Computes the logarithm base 10 of \p x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Log10(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log10)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Log10(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log10)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Log10(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log10)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Log10(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log10)>(x);
}

/// Computes the value of log(1+x) accurately for very small values of x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Log1P(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log1p)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Log1P(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log1p)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Log1P(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log1p)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Log1P(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(log1p)>(x);
}

}
} // dax::math

#endif //__dax_math_Exp_h
