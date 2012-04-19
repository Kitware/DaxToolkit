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
#ifndef __dax_exec_math_Exp_h
#define __dax_exec_math_Exp_h

// This header file defines math functions that deal with exponentials.

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

namespace internal {
struct Inverse
{
  DAX_EXEC_EXPORT dax::Scalar operator()(dax::Scalar x) const { return 1/x; }
};
}

//-----------------------------------------------------------------------------
/// Computes \p x raised to the power of \p y.
///
DAX_EXEC_EXPORT dax::Scalar Pow(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(pow)(x, y);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(sqrt)

/// Compute the square root of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Sqrt(dax::Scalar x) {
  return internal::sqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Sqrt(dax::Vector2 x) {
  return internal::sqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Sqrt(dax::Vector3 x) {
  return internal::sqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Sqrt(dax::Vector4 x) {
  return internal::sqrt_template(x);
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_TEMPLATE(rsqrt)
#else
namespace internal {
template<typename T> DAX_EXEC_EXPORT T rsqrt_template(T x) {
  return dax::exec::VectorMap(dax::exec::math::Sqrt(x), internal::Inverse());
}
}
#endif

/// Compute the reciprocal square root of \p x. The result of this function is
/// equivalent to <tt>1/Sqrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal square root than the regular square root. Thus, you
/// should use this function whenever dividing by the square root.
///
DAX_EXEC_EXPORT dax::Scalar RSqrt(dax::Scalar x) {
  return internal::rsqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 RSqrt(dax::Vector2 x) {
  return internal::rsqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 RSqrt(dax::Vector3 x) {
  return internal::rsqrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 RSqrt(dax::Vector4 x) {
  return internal::rsqrt_template(x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(cbrt)

/// Compute the cube root of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Cbrt(dax::Scalar x) {
  return internal::cbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Cbrt(dax::Vector2 x) {
  return internal::cbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Cbrt(dax::Vector3 x) {
  return internal::cbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Cbrt(dax::Vector4 x) {
  return internal::cbrt_template(x);
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_TEMPLATE(rcbrt)
#else
namespace internal {
template<typename T> DAX_EXEC_EXPORT T rcbrt_template(T x) {
  return dax::exec::VectorMap(dax::exec::math::Cbrt(x), internal::Inverse());
}
}
#endif

/// Compute the reciprocal cube root of \p x. The result of this function is
/// equivalent to <tt>1/Cbrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal cube root than the regular cube root. Thus, you
/// should use this function whenever dividing by the cube root.
///
DAX_EXEC_EXPORT dax::Scalar RCbrt(dax::Scalar x) {
  return internal::rcbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 RCbrt(dax::Vector2 x) {
  return internal::rcbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 RCbrt(dax::Vector3 x) {
  return internal::rcbrt_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 RCbrt(dax::Vector4 x) {
  return internal::rcbrt_template(x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(exp)
DAX_SYS_MATH_TEMPLATE(exp2)
DAX_SYS_MATH_TEMPLATE(expm1)

/// Computes e**\p x, the base-e exponential of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Exp(dax::Scalar x) {
  return internal::exp_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Exp(dax::Vector2 x) {
  return internal::exp_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Exp(dax::Vector3 x) {
  return internal::exp_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Exp(dax::Vector4 x) {
  return internal::exp_template(x);
}

/// Computes 2**\p x, the base-2 exponential of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Exp2(dax::Scalar x) {
  return internal::exp2_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Exp2(dax::Vector2 x) {
  return internal::exp2_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Exp2(dax::Vector3 x) {
  return internal::exp2_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Exp2(dax::Vector4 x) {
  return internal::exp2_template(x);
}

/// Computes (e**\p x) - 1, the of base-e exponental of \p x then minus 1. The
/// accuracy of this function is good even for very small values of x.
///
DAX_EXEC_EXPORT dax::Scalar ExpM1(dax::Scalar x) {
  return internal::expm1_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 ExpM1(dax::Vector2 x) {
  return internal::expm1_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 ExpM1(dax::Vector3 x) {
  return internal::expm1_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 ExpM1(dax::Vector4 x) {
  return internal::expm1_template(x);
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_TEMPLATE(exp10)
#else // ! DAX_CUDA
namespace internal {
struct exp10_functor {
  DAX_EXEC_EXPORT dax::Scalar operator()(dax::Scalar x) const {
    return dax::exec::math::Pow(10, x);
  }
};
template<typename T> DAX_EXEC_EXPORT T exp10_template(T x)
{
  return dax::exec::VectorMap(x, exp10_functor());
}
}
#endif // ! DAX_CUDA

/// Computes 10**\p x, the base-10 exponential of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Exp10(dax::Scalar x) {
  return internal::exp10_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Exp10(dax::Vector2 x) {
  return internal::exp10_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Exp10(dax::Vector3 x) {
  return internal::exp10_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Exp10(dax::Vector4 x) {
  return internal::exp10_template(x);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_TEMPLATE(log)
DAX_SYS_MATH_TEMPLATE(log2)
DAX_SYS_MATH_TEMPLATE(log10)
DAX_SYS_MATH_TEMPLATE(log1p)

/// Computes the natural logarithm of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Log(dax::Scalar x) {
  return internal::log_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Log(dax::Vector2 x) {
  return internal::log_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Log(dax::Vector3 x) {
  return internal::log_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Log(dax::Vector4 x) {
  return internal::log_template(x);
}

/// Computes the logarithm base 2 of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Log2(dax::Scalar x) {
  return internal::log2_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Log2(dax::Vector2 x) {
  return internal::log2_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Log2(dax::Vector3 x) {
  return internal::log2_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Log2(dax::Vector4 x) {
  return internal::log2_template(x);
}

/// Computes the logarithm base 10 of \p x.
///
DAX_EXEC_EXPORT dax::Scalar Log10(dax::Scalar x) {
  return internal::log10_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Log10(dax::Vector2 x) {
  return internal::log10_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Log10(dax::Vector3 x) {
  return internal::log10_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Log10(dax::Vector4 x) {
  return internal::log10_template(x);
}

/// Computes the value of log(1+x) accurately for very small values of x.
///
DAX_EXEC_EXPORT dax::Scalar Log1P(dax::Scalar x) {
  return internal::log1p_template(x);
}
DAX_EXEC_EXPORT dax::Vector2 Log1P(dax::Vector2 x) {
  return internal::log1p_template(x);
}
DAX_EXEC_EXPORT dax::Vector3 Log1P(dax::Vector3 x) {
  return internal::log1p_template(x);
}
DAX_EXEC_EXPORT dax::Vector4 Log1P(dax::Vector4 x) {
  return internal::log1p_template(x);
}

//-----------------------------------------------------------------------------
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_FUNCTION
#undef DAX_SYS_MATH_TEMPLATE

}
}
} // dax::exec::math

#endif //__dax_exec_math_Exp_h
