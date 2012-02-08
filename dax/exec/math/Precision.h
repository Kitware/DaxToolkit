/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_math_Precision_h
#define __dax_exec_math_Precision_h

// This header file defines math functions that deal with the precision of
// floating point numbers.

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>

#ifndef DAX_CUDA
#include <math.h>
#include <limits>
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

//-----------------------------------------------------------------------------
#ifdef DAX_CUDA
#define DAX_USE_IEEE_NONFINITE
#endif

#ifdef DAX_USE_IEEE_NONFINITE

namespace internal {

#if DAX_SIZE_SCALAR == 4
union IEEE754Bits {
  dax::internal::UInt32Type bits;
  dax::Scalar scalar;
};
const IEEE754Bits MathNanBits    = { 0x7FF80000 };
const IEEE754Bits MathInfBits    = { 0x7FF00000 };
const IEEE754Bits MathNegInfBits = { 0xFFF00000 };
#elif DAX_SIZE_SCALAR == 8
union IEEE754Bits {
  dax::internal::UInt64Type bits;
  dax::Scalar scalar;
};
const IEEE754Bits MathNanBits    = { 0x7FF8000000000000LL };
const IEEE754Bits MathInfBits    = { 0x7FF0000000000000LL };
const IEEE754Bits MathNegInfBits = { 0xFFF0000000000000LL };
#else
#error Unknown scalar size
#endif

} // namespace internal

/// Returns the scalar representation for not-a-number (NaN).
///
DAX_EXEC_EXPORT dax::Scalar Nan()
{
  return internal::MathNanBits.scalar;
}

/// Returns the scalar reprentation for infinity.
///
DAX_EXEC_EXPORT dax::Scalar Infinity()
{
  return internal::MathInfBits.scalar;
}

/// Returns the scalar representation for negitive infinity.
///
DAX_EXEC_EXPORT dax::Scalar NegativeInfinity()
{
  return internal::MathNegInfBits.scalar;
}

#else // !DAX_USE_IEEE_NONFINITE

/// Returns the scalar representation for not-a-number (NaN).
///
DAX_EXEC_EXPORT dax::Scalar Nan()
{
  return std::numeric_limits<dax::Scalar>::quiet_NaN();
}

/// Returns the scalar reprentation for infinity.
///
DAX_EXEC_EXPORT dax::Scalar Infinity()
{
  return std::numeric_limits<dax::Scalar>::infinity();
}

/// Returns the scalar representation for negitive infinity.
///
DAX_EXEC_EXPORT dax::Scalar NegativeInfinity()
{
  return -std::numeric_limits<dax::Scalar>::infinity();
}

#endif // !DAX_USE_IEEE_NONFINITE

#ifdef DAX_USE_IEEE_NONFINITE
#undef DAX_USE_IEEE_NONFINITE
#endif

//-----------------------------------------------------------------------------
/// Returns true if \p x is not a number.
///
DAX_EXEC_EXPORT bool IsNan(dax::Scalar x)
{
  return (isnan(x) != 0);
}

/// Returns true if \p is positive or negative infinity.
///
DAX_EXEC_EXPORT bool IsInf(dax::Scalar x)
{
  return (isinf(x) != 0);
}

/// Returns true if \p is a normal number (not NaN or infinite).
///
DAX_EXEC_EXPORT bool IsFinite(dax::Scalar x)
{
  return (isfinite(x) != 0);
}

//-----------------------------------------------------------------------------
/// Computes the remainder on division of 2 floating point numbers. The return
/// value is \p numerator - n \p denominator, where n is the quotient of \p
/// numerator divided by \p denominator rounded towards zero to an integer. For
/// example, <tt>FMod(6.5, 2.3)</tt> returns 1.9, which is 6.5 minus 4.6.
///
template<typename T>
DAX_EXEC_EXPORT T FMod(T numerator, T denominator)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    Traits::SetComponent(result,
                         component,
                         DAX_SYS_MATH_FUNCTION(fmod)(
                           Traits::GetComponent(numerator, component),
                           Traits::GetComponent(denominator, component)));
    }
  return result;
}

//-----------------------------------------------------------------------------
/// Computes the remainder on division of 2 floating point numbers. The return
/// value is \p numerator - n \p denominator, where n is the quotient of \p
/// numerator divided by \p denominator rounded towards the nearest integer
/// (instead of toward zero like FMod). For example, <tt>FMod(6.5, 2.3)</tt>
/// returns -0.4, which is 6.5 minus 6.9.
///
template<typename T>
DAX_EXEC_EXPORT T Remainder(T numerator, T denominator)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    Traits::SetComponent(result,
                         component,
                         DAX_SYS_MATH_FUNCTION(remainder)(
                           Traits::GetComponent(numerator, component),
                           Traits::GetComponent(denominator, component)));
    }
  return result;
}

//-----------------------------------------------------------------------------
/// Returns the remainder on division of 2 floating point numbers just like
/// Remainder. In addition, this function also returns the \c quotient used to
/// get that remainder.
///
template<typename T>
DAX_EXEC_EXPORT T RemainderQuotient(T numerator, T denominator, T &quotient)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    int iQuotient;
    Traits::SetComponent(result,
                         component,
                         DAX_SYS_MATH_FUNCTION(remquo)(
                           Traits::GetComponent(numerator, component),
                           Traits::GetComponent(denominator, component),
                           &iQuotient));
    Traits::SetComponent(quotient, component, dax::Scalar(iQuotient));
    }
  return result;
}

DAX_EXEC_EXPORT dax::Scalar RemainderQuotient(dax::Scalar numerator,
                                              dax::Scalar denominator,
                                              int &quotient)
{
  return DAX_SYS_MATH_FUNCTION(remquo)(numerator, denominator, &quotient);
}

//-----------------------------------------------------------------------------
/// Gets the integral and fractional parts of \c x. The return value is the
/// fractional part and \c integral is set to the integral part.
///
template<typename T>
DAX_EXEC_EXPORT T ModF(T x, T &integral)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    dax::Scalar fractionalPart;
    dax::Scalar integralPart;
    fractionalPart
        = DAX_SYS_MATH_FUNCTION(modf)(Traits::GetComponent(x, component),
                                      &integralPart);
    Traits::SetComponent(result, component, fractionalPart);
    Traits::SetComponent(integral, component, integralPart);
    }
  return result;
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(ceil)
DAX_SYS_MATH_FUNCTOR(floor)
DAX_SYS_MATH_FUNCTOR(round)

/// Round \p x to the smallest integer value not less than x.
///
template<typename T> DAX_EXEC_EXPORT T Ceil(T x)
{
  return dax::exec::VectorMap(x, internal::ceil_functor());
}

/// round \p x to the largest integer value not greater than x.
///
template<typename T> DAX_EXEC_EXPORT T Floor(T x)
{
  return dax::exec::VectorMap(x, internal::floor_functor());
}

/// Round \p x to the nearest integral value.
///
template<typename T> DAX_EXEC_EXPORT T Round(T x)
{
  return dax::exec::VectorMap(x, internal::round_functor());
}

}
}
} // namespace dax::exec::math

#endif //__dax_exec_math_Precision_h
