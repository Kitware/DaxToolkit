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
#ifndef __dax_exec_math_VectorAnalysis_h
#define __dax_exec_math_VectorAnalysis_h

// This header file defines math functions that deal with linear albegra funcitons

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>
#include <dax/exec/math/Sign.h>
#include <dax/exec/math/Exp.h>

#include <dax/internal/MathSystemFunctions.h>

namespace dax {
namespace exec {
namespace math {

// ----------------------------------------------------------------------------
namespace detail {
template <typename T>
DAX_EXEC_CONT_EXPORT dax::Scalar magnitudesquared_template(const T &x)
{
  return dax::dot(x,x);
}
}

/// \brief Returns the square of the magnitude of a vector.
///
/// It is usually much faster to compute the square of the magnitude than the
/// square, so you should use this function in place of Magnitude or RMagnitude
/// when possible.
///
DAX_EXEC_CONT_EXPORT dax::Scalar MagnitudeSquared(dax::Scalar x) {
  return detail::magnitudesquared_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar MagnitudeSquared(dax::Vector2 x) {
  return detail::magnitudesquared_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar MagnitudeSquared(dax::Vector3 x) {
  return detail::magnitudesquared_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar MagnitudeSquared(dax::Vector4 x) {
  return detail::magnitudesquared_template(x);
}

// ----------------------------------------------------------------------------
namespace detail {
template<typename T>
DAX_EXEC_CONT_EXPORT dax::Scalar magnitude_template(const T &x)
{
  return dax::exec::math::Sqrt(magnitudesquared_template(x));
}
}

/// \brief Returns the magnitude of a vector.
///
/// It is usually much faster to compute MagnitudeSquared, so that should be
/// substituted when possible (unless you are just going to take the square
/// root, which would be besides the point). On some hardware it is also faster
/// to find the reciprical magnitude, so RMagnitude should be used if you
/// actually plan to divide by the magnitude.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Magnitude(dax::Scalar x) {
  return dax::exec::math::Abs(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar Magnitude(dax::Vector2 x) {
  return detail::magnitude_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar Magnitude(dax::Vector3 x) {
  return detail::magnitude_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar Magnitude(dax::Vector4 x) {
  return detail::magnitude_template(x);
}

// ----------------------------------------------------------------------------
namespace detail {
template<typename T>
DAX_EXEC_CONT_EXPORT dax::Scalar rmagnitude_template(const T &x)
{
  return dax::exec::math::RSqrt(magnitudesquared_template(x));
}
}

/// \brief Returns the reciprical magnitude of a vector.
///
/// On some hardware RMagnitude is faster than Magnitude, but neither is
/// as fast as MagnitudeSquared.
///
DAX_EXEC_CONT_EXPORT dax::Scalar RMagnitude(dax::Scalar x) {
  return 1/dax::exec::math::Abs(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar RMagnitude(dax::Vector2 x) {
  return detail::rmagnitude_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar RMagnitude(dax::Vector3 x) {
  return detail::rmagnitude_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Scalar RMagnitude(dax::Vector4 x) {
  return detail::rmagnitude_template(x);
}

// ----------------------------------------------------------------------------
namespace detail {
template <typename T>
DAX_EXEC_CONT_EXPORT T normal_template(const T &x ) {
  return dax::exec::math::RSqrt(dax::dot(x,x)) * x;
}
}

/// \brief Returns a normalized version of the given vector.
///
/// The resulting vector points in the same direction but has unit length.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Normal(dax::Scalar x) {
  return detail::normal_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Normal(const dax::Vector2 &x) {
  return detail::normal_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Normal(const dax::Vector3 &x) {
  return detail::normal_template(x);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Normal(const dax::Vector4 &x) {
  return detail::normal_template(x);
}

// ----------------------------------------------------------------------------
/// \brief Changes a vector to be normal.
///
/// The given vector is scaled to be unit length.
///
DAX_EXEC_CONT_EXPORT void Normalize(dax::Scalar &x) {
  x = Normal(x);
}
DAX_EXEC_CONT_EXPORT void Normalize(dax::Vector2 &x) {
  x = Normal(x);
}
DAX_EXEC_CONT_EXPORT void Normalize(dax::Vector3 &x) {
  x = Normal(x);
}
DAX_EXEC_CONT_EXPORT void Normalize(dax::Vector4 &x) {
  x = Normal(x);
}

// ----------------------------------------------------------------------------
/// \brief Find the cross product of two vectors.
///
DAX_EXEC_CONT_EXPORT dax::Vector3 Cross(const dax::Vector3 &x, const dax::Vector3 &y)
{
  return dax::make_Vector3(x[1]*y[2] - x[2]*y[1],
                           x[2]*y[0] - x[0]*y[2],
                           x[0]*y[1] - x[1]*y[0]);
}

//-----------------------------------------------------------------------------
/// \brief Find the normal of a triangle.
///
/// Given three coordinates in space, which, unless degenerate, uniquely define
/// a triangle and the plane the triangle is on, returns a vector perpendicular
/// to that triangle/plane.
///
DAX_EXEC_CONT_EXPORT dax::Vector3 TriangleNormal(const dax::Vector3 &a,
                                            const dax::Vector3 &b,
                                            const dax::Vector3 &c)
{
  return dax::exec::math::Cross(b-a, c-a);
}

}
}
} // namespace dax::exec::math

#endif //__dax_exec_math_VectorAnalysis_h
