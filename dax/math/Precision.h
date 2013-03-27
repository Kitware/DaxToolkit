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
#ifndef __dax_math_Precision_h
#define __dax_math_Precision_h

// This header file defines math functions that deal with the precision of
// floating point numbers.

#include <dax/internal/MathSystemFunctions.h>

#if _WIN32 && !defined DAX_CUDA_COMPILATION
#include <boost/math/special_functions/round.hpp>
#define DAX_USE_WIN_MATH
#endif

#ifndef DAX_CUDA
#include <limits>

// These nonfinite test functions are usually defined as macros, and boost
// seems to want to undefine those macros so that it can implement the C99
// templates and other implementations of the same name. Get around the problem
// by using the boost version when compiling for a CPU.
#include <boost/math/special_functions/fpclassify.hpp>
#include <cmath>
#define DAX_USE_BOOST_CLASSIFY

#endif

namespace dax {
namespace math {

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
#define DAX_NAN_BITS      0x7FC00000
#define DAX_INF_BITS      0x7F800000
#define DAX_NEG_INF_BITS  0xFF800000
#define DAX_EPSILON       1e-5f
#elif DAX_SIZE_SCALAR == 8
union IEEE754Bits {
  dax::internal::UInt64Type bits;
  dax::Scalar scalar;
};
#define DAX_NAN_BITS      0x7FF8000000000000LL
#define DAX_INF_BITS      0x7FF0000000000000LL
#define DAX_NEG_INF_BITS  0xFFF0000000000000LL
#define DAX_EPSILON       1e-9
#else
#error Unknown scalar size
#endif

} // namespace internal

#define DAX_DEFINE_BITS(name, rep) \
  const dax::math::internal::IEEE754Bits name = { rep }

/// Returns the scalar representation for not-a-number (NaN).
///
DAX_EXEC_CONT_EXPORT dax::Scalar Nan()
{
  DAX_DEFINE_BITS(NanBits, DAX_NAN_BITS);
  return NanBits.scalar;
}

/// Returns the scalar reprentation for infinity.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Infinity()
{
  DAX_DEFINE_BITS(InfBits, DAX_INF_BITS);
  return InfBits.scalar;
}

/// Returns the scalar representation for negitive infinity.
///
DAX_EXEC_CONT_EXPORT dax::Scalar NegativeInfinity()
{
  DAX_DEFINE_BITS(NegInfBits, DAX_NEG_INF_BITS);
  return NegInfBits.scalar;
}

/// Returns the difference between 1 and the least value greater than 1
/// that is representable.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Epsilon()
{
  return DAX_EPSILON;
}

#else // !DAX_USE_IEEE_NONFINITE

/// Returns the scalar representation for not-a-number (NaN).
///
DAX_EXEC_CONT_EXPORT dax::Scalar Nan()
{
  return std::numeric_limits<dax::Scalar>::quiet_NaN();
}

/// Returns the scalar reprentation for infinity.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Infinity()
{
  return std::numeric_limits<dax::Scalar>::infinity();
}

/// Returns the scalar representation for negitive infinity.
///
DAX_EXEC_CONT_EXPORT dax::Scalar NegativeInfinity()
{
  return -std::numeric_limits<dax::Scalar>::infinity();
}

/// Returns the difference between 1 and the least value greater than 1
/// that is representable.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Epsilon()
{
  return std::numeric_limits<dax::Scalar>::epsilon();
}

#endif // !DAX_USE_IEEE_NONFINITE

#ifdef DAX_USE_IEEE_NONFINITE
#undef DAX_USE_IEEE_NONFINITE
#undef DAX_DEFINE_BITS
#undef DAX_NAN_BITS
#undef DAX_INF_BITS
#undef DAX_NEG_INF_BITS
#endif

//-----------------------------------------------------------------------------
/// Returns true if \p x is not a number.
///
DAX_EXEC_CONT_EXPORT bool IsNan(dax::Scalar x)
{
#ifdef DAX_USE_BOOST_CLASSIFY
  using boost::math::isnan;
#endif
  return (isnan(x) != 0);
}

/// Returns true if \p is positive or negative infinity.
///
DAX_EXEC_CONT_EXPORT bool IsInf(dax::Scalar x)
{
#ifdef DAX_USE_BOOST_CLASSIFY
  using boost::math::isinf;
#endif
  return (isinf(x) != 0);
}

/// Returns true if \p is a normal number (not NaN or infinite).
///
DAX_EXEC_CONT_EXPORT bool IsFinite(dax::Scalar x)
{
#ifdef DAX_USE_BOOST_CLASSIFY
  using boost::math::isfinite;
#endif
  return (isfinite(x) != 0);
}

//-----------------------------------------------------------------------------
namespace internal {
template<typename T>
DAX_EXEC_CONT_EXPORT T fmod_template(T numerator, T denominator)
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
}

/// Computes the remainder on division of 2 floating point numbers. The return
/// value is \p numerator - n \p denominator, where n is the quotient of \p
/// numerator divided by \p denominator rounded towards zero to an integer. For
/// example, <tt>FMod(6.5, 2.3)</tt> returns 1.9, which is 6.5 minus 4.6.
///
DAX_EXEC_CONT_EXPORT
dax::Scalar FMod(dax::Scalar numerator, dax::Scalar denominator) {
  return internal::fmod_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector2 FMod(dax::Vector2 numerator, dax::Vector2 denominator) {
  return internal::fmod_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector3 FMod(dax::Vector3 numerator, dax::Vector3 denominator) {
  return internal::fmod_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector4 FMod(dax::Vector4 numerator, dax::Vector4 denominator) {
  return internal::fmod_template(numerator, denominator);
}

//-----------------------------------------------------------------------------
namespace internal {
template<typename T>
DAX_EXEC_CONT_EXPORT T remainder_template(T numerator, T denominator)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
#ifdef DAX_USE_WIN_MATH
    const int quotient = static_cast<int>(boost::math::round(
        Traits::GetComponent(numerator, component) /
        Traits::GetComponent(denominator, component)));

    const dax::Scalar rem = Traits::GetComponent(numerator, component) - 
                            (quotient * Traits::GetComponent(denominator, component));

    Traits::SetComponent(result,
                         component,
                         rem);
#else
    Traits::SetComponent(result,
                         component,
                         DAX_SYS_MATH_FUNCTION(remainder)(
                           Traits::GetComponent(numerator, component),
                           Traits::GetComponent(denominator, component)));
#endif
    }
  return result;
}
}

/// Computes the remainder on division of 2 floating point numbers. The return
/// value is \p numerator - n \p denominator, where n is the quotient of \p
/// numerator divided by \p denominator rounded towards the nearest integer
/// (instead of toward zero like FMod). For example, <tt>FMod(6.5, 2.3)</tt>
/// returns -0.4, which is 6.5 minus 6.9.
///
DAX_EXEC_CONT_EXPORT
dax::Scalar Remainder(dax::Scalar numerator, dax::Scalar denominator) {
  return internal::remainder_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector2 Remainder(dax::Vector2 numerator, dax::Vector2 denominator) {
  return internal::remainder_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector3 Remainder(dax::Vector3 numerator, dax::Vector3 denominator) {
  return internal::remainder_template(numerator, denominator);
}
DAX_EXEC_CONT_EXPORT
dax::Vector4 Remainder(dax::Vector4 numerator, dax::Vector4 denominator) {
  return internal::remainder_template(numerator, denominator);
}

//-----------------------------------------------------------------------------
namespace internal {
template<typename T>
DAX_EXEC_CONT_EXPORT T remquo_template(T numerator, T denominator, T &quotient)
{
  typedef dax::VectorTraits<T> Traits;
  T result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    int iQuotient;
#ifdef DAX_USE_WIN_MATH
    iQuotient = static_cast<int>(boost::math::round(
      Traits::GetComponent(numerator, component) /
      Traits::GetComponent(denominator, component)) );
    
    Traits::SetComponent(result,
                         component,
                         dax::math::Remainder(
                                  Traits::GetComponent(numerator, component),
                                  Traits::GetComponent(denominator, component))
                         );
#else
    Traits::SetComponent(result,
                         component,
                         DAX_SYS_MATH_FUNCTION(remquo)(
                           Traits::GetComponent(numerator, component),
                           Traits::GetComponent(denominator, component),
                           &iQuotient));
#endif
    Traits::SetComponent(quotient, component, dax::Scalar(iQuotient));
    }
  return result;
}
}

/// Returns the remainder on division of 2 floating point numbers just like
/// Remainder. In addition, this function also returns the \c quotient used to
/// get that remainder.
///
DAX_EXEC_CONT_EXPORT dax::Scalar RemainderQuotient(dax::Scalar numerator,
                                              dax::Scalar denominator,
                                              int &quotient)
{
#ifdef DAX_USE_WIN_MATH
  quotient = static_cast<int>(boost::math::round(numerator/denominator));
  return dax::math::Remainder(numerator,denominator);
#else
  return DAX_SYS_MATH_FUNCTION(remquo)(numerator, denominator, &quotient);
#endif
  
}
DAX_EXEC_CONT_EXPORT dax::Scalar RemainderQuotient(dax::Scalar numerator,
                                              dax::Scalar denominator,
                                              dax::Scalar &quotient)
{
  return internal::remquo_template(numerator, denominator, quotient);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 RemainderQuotient(dax::Vector2 numerator,
                                               dax::Vector2 denominator,
                                               dax::Vector2 &quotient)
{
  return internal::remquo_template(numerator, denominator, quotient);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 RemainderQuotient(dax::Vector3 numerator,
                                               dax::Vector3 denominator,
                                               dax::Vector3 &quotient)
{
  return internal::remquo_template(numerator, denominator, quotient);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 RemainderQuotient(dax::Vector4 numerator,
                                               dax::Vector4 denominator,
                                               dax::Vector4 &quotient)
{
  return internal::remquo_template(numerator, denominator, quotient);
}

//-----------------------------------------------------------------------------
namespace internal {
template<typename T>
DAX_EXEC_CONT_EXPORT T modf_template(T x, T &integral)
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
}

/// Gets the integral and fractional parts of \c x. The return value is the
/// fractional part and \c integral is set to the integral part.
///
DAX_EXEC_CONT_EXPORT dax::Scalar ModF(dax::Scalar x, dax::Scalar &integral)
{
  return internal::modf_template(x, integral);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 ModF(dax::Vector2 x, dax::Vector2 &integral)
{
  return internal::modf_template(x, integral);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 ModF(dax::Vector3 x, dax::Vector3 &integral)
{
  return internal::modf_template(x, integral);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 ModF(dax::Vector4 x, dax::Vector4 &integral)
{
  return internal::modf_template(x, integral);
}

//-----------------------------------------------------------------------------
/// Round \p x to the smallest integer value not less than x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Ceil(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(ceil)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Ceil(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(ceil)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Ceil(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(ceil)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Ceil(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(ceil)>(x);
}

/// round \p x to the largest integer value not greater than x.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Floor(dax::Scalar x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(floor)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Floor(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(floor)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Floor(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(floor)>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Floor(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(floor)>(x);
}

/// Round \p x to the nearest integral value.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Round(dax::Scalar x) {
#ifdef DAX_USE_WIN_MATH
  //windows doesn't have any of the c99 math functions
  return boost::math::round(x);
#else
  return DAX_SYS_MATH_FUNCTION(round)(x);
#endif
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Round(dax::Vector2 x) {
  return dax::internal::SysMathVectorCall<dax::math::Round>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Round(dax::Vector3 x) {
  return dax::internal::SysMathVectorCall<dax::math::Round>(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Round(dax::Vector4 x) {
  return dax::internal::SysMathVectorCall<dax::math::Round>(x);
}

}
} // namespace dax::math

#ifdef DAX_USE_BOOST_CLASSIFY
#undef DAX_USE_BOOST_CLASSIFY
#endif

#endif //__dax_math_Precision_h
