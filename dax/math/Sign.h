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
#ifndef __dax_math_Sign_h
#define __dax_math_Sign_h

// This header file defines math functions that deal with the sign (positive or
// negative) of numbers.
#include <dax/TypeTraits.h>
#include <dax/internal/MathSystemFunctions.h>

#ifndef DAX_CUDA

#include <stdlib.h>

// signbit is usually defined as a macro, and boost seems to want to undefine
// that macro so that it can implement the C99 templates and other
// implementations of the same name. Get around the problem by using the boost
// version when compiling for a CPU.
#include <boost/math/special_functions/sign.hpp>
#define DAX_USE_BOOST_SIGN

#endif

namespace dax {
namespace math {

//forward declare abs signature so detail can call it
template<class T> DAX_EXEC_CONT_EXPORT T Abs(const T& x);

namespace detail{
template<class NumericTag> struct Abs {
  template<class T>
  DAX_EXEC_CONT_EXPORT
  T operator()(const T& x) const
  {
    return dax::internal::SysMathVectorCall<DAX_SYS_MATH_FUNCTION(fabs)>(x);
  }
};

template<> struct Abs<dax::TypeTraitsIntegerTag>
{
  template<class ValueType>
  DAX_EXEC_CONT_EXPORT
  ValueType operator()( ValueType x ) const
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

  DAX_EXEC_CONT_EXPORT
  dax::Id3 operator()(const dax::Id3& x) const
  { return dax::make_Id3(dax::math::Abs(x[0]),
                         dax::math::Abs(x[1]),
                         dax::math::Abs(x[2])); }
};


}


//-----------------------------------------------------------------------------
/// Return the absolute value of \x. That is, return \p x if it is positive or
/// \p -x if it is negative.
///
template<class T>
DAX_EXEC_CONT_EXPORT T Abs(const T& x) {
  typedef typename dax::TypeTraits<T> TTraits;
  return detail::Abs<typename TTraits::NumericTag>()(x);
}

//-----------------------------------------------------------------------------
/// Returns true if \p x is less than zero, false otherwise.
///
DAX_EXEC_CONT_EXPORT bool IsNegative(dax::Scalar x)
{
#ifdef DAX_USE_BOOST_SIGN
  using boost::math::signbit;
#endif
  return (signbit(x) != 0);
}

//-----------------------------------------------------------------------------
/// Returns a nonzero value if \x is negative.
///
DAX_EXEC_CONT_EXPORT int SignBit(dax::Scalar x)
{
#ifdef DAX_USE_BOOST_SIGN
  using boost::math::signbit;
#endif
  return signbit(x);
}

//-----------------------------------------------------------------------------
/// Copies the sign of \p y onto \p x.  If \p y is positive, returns Abs(\p x).
/// If \p y is negative, returns -Abs(\p x).
///
DAX_EXEC_CONT_EXPORT dax::Scalar CopySign(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(copysign)(x, y);
}

}
} // namespace dax::math

#ifdef DAX_USE_BOOST_SIGN
#undef DAX_USE_BOOST_SIGN
#endif

#endif //__dax_math_Sign_h
