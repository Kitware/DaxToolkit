/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
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
DAX_SYS_MATH_FUNCTOR(sqrt)

/// Compute the square root of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Sqrt(T x)
{
  return dax::exec::VectorMap(x, internal::sqrt_functor());
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_FUNCTOR(rsqrt)

/// Compute the reciprocal square root of \p x. The result of this function is
/// equivalent to <tt>1/Sqrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal square root than the regular square root. Thus, you
/// should use this function whenever dividing by the square root.
///
template<typename T> DAX_EXEC_EXPORT T RSqrt(T x)
{
  return dax::exec::VectorMap(x, internal::rsqrt_functor());
}
#else // !DAX_CUDA

/// Compute the reciprocal square root of \p x. The result of this function is
/// equivalent to <tt>1/Sqrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal square root than the regular square root. Thus, you
/// should use this function whenever dividing by the square root.
///
template<typename T> DAX_EXEC_EXPORT T RSqrt(T x) {
  return dax::exec::VectorMap(Sqrt(x), internal::Inverse());
}
#endif // !DAX_CUDA

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(cbrt)

/// Compute the cube root of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Cbrt(T x)
{
  return dax::exec::VectorMap(x, internal::cbrt_functor());
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_FUNCTOR(rcbrt)

/// Compute the reciprocal cube root of \p x. The result of this function is
/// equivalent to <tt>1/Cbrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal cube root than the regular cube root. Thus, you
/// should use this function whenever dividing by the cube root.
///
template<typename T> DAX_EXEC_EXPORT T RCbrt(T x)
{
  return dax::exec::VectorMap(x, internal::rsqrt_functor());
}
#else // !DAX_CUDA

/// Compute the reciprocal cube root of \p x. The result of this function is
/// equivalent to <tt>1/Cbrt(x)</tt>. However, on some devices it is faster to
/// compute the reciprocal cube root than the regular cube root. Thus, you
/// should use this function whenever dividing by the cube root.
///
template<typename T> DAX_EXEC_EXPORT T RCbrt(T x) {
  return dax::exec::VectorMap(Cbrt(x), internal::Inverse());
}
#endif // !DAX_CUDA

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(exp)
DAX_SYS_MATH_FUNCTOR(exp2)
DAX_SYS_MATH_FUNCTOR(expm1)

/// Computes e**\p x, the base-e exponential of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Exp(T x)
{
  return dax::exec::VectorMap(x, internal::exp_functor());
}

/// Computes 2**\p x, the base-2 exponential of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Exp2(T x)
{
  return dax::exec::VectorMap(x, internal::exp2_functor());
}

/// Computes (e**\p x) - 1, the of base-e exponental of \p x then minus 1. The
/// accuracy of this function is good even for very small values of x.
///
template<typename T> DAX_EXEC_EXPORT T ExpM1(T x)
{
  return dax::exec::VectorMap(x, internal::expm1_functor());
}

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
DAX_SYS_MATH_FUNCTOR(exp10)
#else // ! DAX_CUDA
namespace internal {
struct exp10_functor {
  DAX_EXEC_EXPORT dax::Scalar operator()(dax::Scalar x) const {
    return dax::exec::math::Pow(10, x);
  }
};
}
#endif // ! DAX_CUDA

/// Computes 10**\p x, the base-10 exponential of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Exp10(T x)
{
  return dax::exec::VectorMap(x, internal::exp10_functor());
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(log)
DAX_SYS_MATH_FUNCTOR(log2)
DAX_SYS_MATH_FUNCTOR(log10)
DAX_SYS_MATH_FUNCTOR(log1p)

/// Computes the natural logarithm of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Log(T x)
{
  return dax::exec::VectorMap(x, internal::log_functor());
}

/// Computes the logarithm base 2 of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Log2(T x)
{
  return dax::exec::VectorMap(x, internal::log2_functor());
}

/// Computes the logarithm base 10 of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Log10(T x)
{
  return dax::exec::VectorMap(x, internal::log10_functor());
}

/// Computes the value of log(1+x) accurately for very small values of x.
///
template<typename T> DAX_EXEC_EXPORT T Log1P(T x)
{
  return dax::exec::VectorMap(x, internal::log1p_functor());
}

//-----------------------------------------------------------------------------
#undef DAX_SYS_MATH_FUNCTOR
#undef DAX_SYS_MATH_FUNCTION

}
}
} // dax::exec::math

#endif //__dax_exec_math_Exp_h
