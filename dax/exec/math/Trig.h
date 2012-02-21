/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
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

//-----------------------------------------------------------------------------
/// Returns the constant Pi.
///
DAX_EXEC_EXPORT dax::Scalar Pi()
{
  return dax::Scalar(3.14159265358979323846264338327950288);
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(sin)
DAX_SYS_MATH_FUNCTOR(cos)
DAX_SYS_MATH_FUNCTOR(tan)

/// Compute the sine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Sin(T x)
{
  return dax::exec::VectorMap(x, internal::sin_functor());
}

/// Compute the cosine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Cos(T x)
{
  return dax::exec::VectorMap(x, internal::cos_functor());
}

/// Compute the tangent of \p x.
///
template<typename T> DAX_EXEC_EXPORT T Tan(T x)
{
  return dax::exec::VectorMap(x, internal::tan_functor());
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(asin)
DAX_SYS_MATH_FUNCTOR(acos)
DAX_SYS_MATH_FUNCTOR(atan)

/// Compute the arc sine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ASin(T x)
{
  return dax::exec::VectorMap(x, internal::asin_functor());
}

/// Compute the arc cosine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ACos(T x)
{
  return dax::exec::VectorMap(x, internal::acos_functor());
}

/// Compute the arc tangent of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ATan(T x)
{
  return dax::exec::VectorMap(x, internal::atan_functor());
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
DAX_SYS_MATH_FUNCTOR(sinh)
DAX_SYS_MATH_FUNCTOR(cosh)
DAX_SYS_MATH_FUNCTOR(tanh)

/// Compute the hyperbolic sine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T SinH(T x)
{
  return dax::exec::VectorMap(x, internal::sinh_functor());
}

/// Compute the hyperbolic cosine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T CosH(T x)
{
  return dax::exec::VectorMap(x, internal::cosh_functor());
}

/// Compute the hyperbolic tangent of \p x.
///
template<typename T> DAX_EXEC_EXPORT T TanH(T x)
{
  return dax::exec::VectorMap(x, internal::tanh_functor());
}

//-----------------------------------------------------------------------------
DAX_SYS_MATH_FUNCTOR(asinh)
DAX_SYS_MATH_FUNCTOR(acosh)
DAX_SYS_MATH_FUNCTOR(atanh)

/// Compute the hyperbolic arc sine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ASinH(T x)
{
  return dax::exec::VectorMap(x, internal::asinh_functor());
}

/// Compute the hyperbolic arc cosine of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ACosH(T x)
{
  return dax::exec::VectorMap(x, internal::acosh_functor());
}

/// Compute the hyperbolic arc tangent of \p x.
///
template<typename T> DAX_EXEC_EXPORT T ATanH(T x)
{
  return dax::exec::VectorMap(x, internal::atanh_functor());
}

}
}
} // namespace dax::exec::math

#endif //__dax_exec_math_Trig_h
