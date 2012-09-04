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
#ifndef __dax_math_Compare_h
#define __dax_math_Compare_h

//needed to for proper specialization for dax tuple comparisons
#include <dax/TypeTraits.h>
#include <dax/VectorTraits.h>

// This header file defines math functions that do comparisons.
#include <dax/internal/MathSystemFunctions.h>

namespace dax {
namespace math {


//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is larger.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Max(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(fmax)(x, y);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Max(dax::Vector2 x, dax::Vector2 y)
{
  return dax::make_Vector2(Max(x[0], y[0]), Max(x[1], y[1]));
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Max(dax::Vector3 x, dax::Vector3 y)
{
  return dax::make_Vector3(Max(x[0], y[0]), Max(x[1], y[1]), Max(x[2], y[2]));
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Max(dax::Vector4 x, dax::Vector4 y)
{
  return dax::make_Vector4(Max(x[0], y[0]),
                           Max(x[1], y[1]),
                           Max(x[2], y[2]),
                           Max(x[3], y[3]));
}
DAX_EXEC_CONT_EXPORT dax::Id Max(dax::Id x, dax::Id y)
{
  return (x < y) ? y : x;
}
DAX_EXEC_CONT_EXPORT dax::Id3 Max(dax::Id3 x, dax::Id3 y)
{
  return dax::make_Id3(Max(x[0], y[0]), Max(x[1], y[1]), Max(x[2], y[2]));
}

//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is smaller.
///
DAX_EXEC_CONT_EXPORT dax::Scalar Min(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(fmin)(x, y);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 Min(dax::Vector2 x, dax::Vector2 y)
{
  return dax::make_Vector2(Min(x[0], y[0]), Min(x[1], y[1]));
}
DAX_EXEC_CONT_EXPORT dax::Vector3 Min(dax::Vector3 x, dax::Vector3 y)
{
  return dax::make_Vector3(Min(x[0], y[0]), Min(x[1], y[1]), Min(x[2], y[2]));
}
DAX_EXEC_CONT_EXPORT dax::Vector4 Min(dax::Vector4 x, dax::Vector4 y)
{
  return dax::make_Vector4(Min(x[0], y[0]),
                           Min(x[1], y[1]),
                           Min(x[2], y[2]),
                           Min(x[3], y[3]));
}
DAX_EXEC_CONT_EXPORT dax::Id Min(dax::Id x, dax::Id y)
{
  return (x < y) ? x : y;
}
DAX_EXEC_CONT_EXPORT dax::Id3 Min(dax::Id3 x, dax::Id3 y)
{
  return dax::make_Id3(Min(x[0], y[0]), Min(x[1], y[1]), Min(x[2], y[2]));
}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares equal to the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is equal to the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct Equal
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    return a == b;
  }
};

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares not equal to the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is not qual to the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct NotEqual
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    return a != b;
  }
};

// ----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct greater_equal {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return dax::math::detail::greater_equal<Dimensionality,Size-1>()(a,b) && a[Size-1] >= b[Size-1]; }
};
template<> struct greater_equal<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] >= b[0]; }
};
template<> struct greater_equal<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a >= b; }
};
}


//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares greater than or equal to the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is greater than or equal to the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct GreaterEqual
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::greater_equal<Dimensionality,SIZE>()(a,b);
  }
};


//-----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct greater {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return dax::math::detail::greater<Dimensionality,Size-1>()(a,b) && a[Size-1] > b[Size-1]; }
};
template<> struct greater<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] > b[0]; }
};
template<> struct greater<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a > b; }
};

}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares greater than the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is greater than or the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct Greater
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::greater<Dimensionality,SIZE>()(a,b);
  }
};

//-----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct less_equal {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return dax::math::detail::less_equal<Dimensionality,Size-1>()(a,b) && a[Size-1] <= b[Size-1]; }
};
template<> struct less_equal<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] <= b[0]; }
};
template<> struct less_equal<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a <= b; }
};
}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares less than or equal to the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is less than or equal to the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct LessEqual
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::less_equal<Dimensionality,SIZE>()(a,b);
  }
};

//-----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct less {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return dax::math::detail::less<Dimensionality,Size-1>()(a,b) && a[Size-1] < b[Size-1]; }
};
template<> struct less<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] < b[0];}
};
template<> struct less<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a < b; }
};
}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares less than the second \p item, and false otherwise.
/// In the case of dax::Tuples and other vector type (dax::Vector3 etc)
/// comparsion verifies than each component in the vector
/// is less than or the same component in the second vector
///
/// This is a binary function to allow it to be applied using dax::VectorOperations
/// Note: dax::Tuple of dax::Tuples currently aren't supported.
struct Less
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::less<Dimensionality,SIZE>()(a,b);
  }
};

}
} // namespace dax::math

#endif //__dax_math_Compare_h
