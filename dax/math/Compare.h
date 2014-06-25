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


// This header file defines math functions that do comparisons.
#include <dax/internal/MathSystemFunctions.h>

//if we are compiling with VS2012 and above we support fminf and fmaxf
#if _MSC_VER <= 1600 && !defined DAX_CUDA_COMPILATION
  #define DAX_USE_STL_MIN_MAX
  #include <algorithm>
#endif

#ifdef DAX_ENABLE_THRUST
  //forward declare thrust::device_reference
  namespace thrust { template<typename T> class device_reference; }
#endif

namespace dax {
namespace math {

//forward declare max function
template<class T> DAX_EXEC_CONT_EXPORT T Max(const T& x, const T& y);

namespace detail{
template<class NumericTag, class DimTag> struct max {

  DAX_EXEC_CONT_EXPORT
  dax::Vector2 operator()(const dax::Vector2& x, const dax::Vector2& y) const
  { return dax::make_Vector2(dax::math::Max(x[0], y[0]),
                             dax::math::Max(x[1], y[1])); }

  DAX_EXEC_CONT_EXPORT
  dax::Vector3 operator()(const dax::Vector3& x, const dax::Vector3& y) const
  { return dax::make_Vector3(dax::math::Max(x[0], y[0]),
                             dax::math::Max(x[1], y[1]),
                             dax::math::Max(x[2], y[2])); }

  DAX_EXEC_CONT_EXPORT
  dax::Vector4 operator()(const dax::Vector4& x, const dax::Vector4& y) const
  { return dax::make_Vector4(dax::math::Max(x[0], y[0]),
                             dax::math::Max(x[1], y[1]),
                             dax::math::Max(x[2], y[2]),
                             dax::math::Max(x[3], y[3])); }

  DAX_EXEC_CONT_EXPORT
  dax::Id3 operator()(const dax::Id3& x, const dax::Id3& y) const
  { return dax::make_Id3(dax::math::Max(x[0], y[0]),
                         dax::math::Max(x[1], y[1]),
                         dax::math::Max(x[2], y[2])); }

  template<class T>
  DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  { typedef dax::VectorTraits<T> VTraits;
    enum{SIZE=VTraits::NUM_COMPONENTS};
    T result;
    for(int i=0;i<SIZE;++i)
      {
      result[i]=dax::math::Max(x[i],y[i]);
      }
    return result;
    }
};

template<> struct max<dax::TypeTraitsIntegerTag,dax::TypeTraitsScalarTag> {
  template<class T>
  DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  { return (x < y) ? y : x; }
};

template<> struct max<dax::TypeTraitsRealTag,dax::TypeTraitsScalarTag> {
  template<class T>
  DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  {
#ifdef DAX_USE_STL_MIN_MAX
  return (std::max)(x, y); //wrap in () to avoid window.h max macro
#else
  return DAX_SYS_MATH_FUNCTION(fmax)(x, y);
#endif
  }
};

}


//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is larger.
///
template<class T>
DAX_EXEC_CONT_EXPORT T Max(const T& x, const T& y)
{
  typedef typename dax::TypeTraits<T> TTraits;
  return detail::max<typename TTraits::NumericTag,
                     typename TTraits::DimensionalityTag>()(x,y);
}

//forward declare max function
template<class T> DAX_EXEC_CONT_EXPORT T Min(const T& x, const T& y);

namespace detail{
template<class NumericTag, class DimTag> struct min {

  DAX_EXEC_CONT_EXPORT
  dax::Vector2 operator()(const dax::Vector2& x, const dax::Vector2& y) const
  { return dax::make_Vector2(dax::math::Min(x[0], y[0]),
                             dax::math::Min(x[1], y[1])); }

  DAX_EXEC_CONT_EXPORT
  dax::Vector3 operator()(const dax::Vector3& x, const dax::Vector3& y) const
  { return dax::make_Vector3(dax::math::Min(x[0], y[0]),
                             dax::math::Min(x[1], y[1]),
                             dax::math::Min(x[2], y[2])); }

  DAX_EXEC_CONT_EXPORT
  dax::Vector4 operator()(const dax::Vector4& x, const dax::Vector4& y) const
  { return dax::make_Vector4(dax::math::Min(x[0], y[0]),
                             dax::math::Min(x[1], y[1]),
                             dax::math::Min(x[2], y[2]),
                             dax::math::Min(x[3], y[3])); }

  DAX_EXEC_CONT_EXPORT
  dax::Id3 operator()(const dax::Id3& x, const dax::Id3& y) const
  { return dax::make_Id3(dax::math::Min(x[0], y[0]),
                         dax::math::Min(x[1], y[1]),
                         dax::math::Min(x[2], y[2])); }

  template<class T>
  DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  { typedef dax::VectorTraits<T> VTraits;
    enum{SIZE=VTraits::NUM_COMPONENTS};
    T result;
    for(int i=0;i<SIZE;++i)
      {
      result[i]=dax::math::Min(x[i],y[i]);
      }
    return result;
    }
};

template<> struct min<dax::TypeTraitsIntegerTag,dax::TypeTraitsScalarTag> {
  template<class T>
  DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  { return (x < y) ? x : y; }
};

template<> struct min<dax::TypeTraitsRealTag,dax::TypeTraitsScalarTag> {
  template<class T>
 DAX_EXEC_CONT_EXPORT T operator()(const T& x, const T& y) const
  {
#ifdef DAX_USE_STL_MIN_MAX
  return (std::min)(x, y); //wrap in () to avoid window.h min macro
#else
  return DAX_SYS_MATH_FUNCTION(fmin)(x, y);
#endif
  }
};

}

//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is smaller.
///
template<class T>
DAX_EXEC_CONT_EXPORT T Min(const T& x, const T& y)
{
  typedef typename dax::TypeTraits<T> TTraits;
  return detail::min<typename TTraits::NumericTag,
                     typename TTraits::DimensionalityTag>()(x,y);
}

//-----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct sort_greater {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    for(dax::Id i=0; i < Size; ++i)
    {
    //ignore equals as that represents check next value
    if(a[i] > b[i])
      return true;
    else if(a[i] < b[i])
      return false;
    }
  //this will be hit if a equals b exactly
  return false;
  }
};
template<> struct sort_greater<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] > b[0]; }
};
template<> struct sort_greater<dax::TypeTraitsVectorTag,2> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] > b[0]) ||
           (a[0] == b[0] && a[1] > b[1]);
  }
};
template<> struct sort_greater<dax::TypeTraitsVectorTag,3> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] > b[0]) ||
           (a[0] == b[0] && a[1] > b[1]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] > b[2]) ;
  }
};
template<> struct sort_greater<dax::TypeTraitsVectorTag,4> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] > b[0]) ||
           (a[0] == b[0] && a[1] > b[1]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] > b[2]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] > b[3]);
  }
};
template<> struct sort_greater<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a > b; }
};
}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares greater than the second \p item,
/// and false otherwise. Uses an ordered sorting comparison function to
/// allow people to use this functor with algorithms like Sort, Unique, LowerBounds.
struct SortGreater
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::sort_greater<Dimensionality,SIZE>()(a,b);
  }

#ifdef DAX_ENABLE_THRUST
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,
                                   const ::thrust::device_reference<T> b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::sort_greater<Dimensionality,SIZE>()(a,(T)b);
  }
#endif
};

//-----------------------------------------------------------------------------
namespace detail {
template<typename Dimensionality,int Size> struct sort_less {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    for(dax::Id i=0; i < Size; ++i)
    {
    //ignore equals as that represents check next value
    if(a[i] < b[i])
      return true;
    else if(a[i] > b[i])
      return false;
    }
  //this will be hit if a equals b exactly
  return false;
  }
};
template<> struct sort_less<dax::TypeTraitsVectorTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a[0] < b[0]; }
};
template<> struct sort_less<dax::TypeTraitsVectorTag,2> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] < b[0]) ||
           (a[0] == b[0] && a[1] < b[1]);
  }
};
template<> struct sort_less<dax::TypeTraitsVectorTag,3> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] < b[0]) ||
           (a[0] == b[0] && a[1] < b[1]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]) ;
  }
};
template<> struct sort_less<dax::TypeTraitsVectorTag,4> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return (a[0] < b[0]) ||
           (a[0] == b[0] && a[1] < b[1]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] < b[2]) ||
           (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] < b[3]);
  }
};
template<> struct sort_less<dax::TypeTraitsScalarTag,1> {
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  { return a < b; }
};
}

//-----------------------------------------------------------------------------
/// Returns true if its first \p item compares less than the second \p item,
/// and false otherwise. Uses an ordered sorting comparison function to
/// allow people to use this functor with algorithms like Sort, Unique, LowerBounds.
struct SortLess
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,const T& b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::sort_less<Dimensionality,SIZE>()(a,b);
  }

#ifdef DAX_ENABLE_THRUST
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a,
                                   const ::thrust::device_reference<T> b) const
  {
    typedef typename dax::TypeTraits<T>::DimensionalityTag Dimensionality;
    enum{SIZE = dax::VectorTraits<T>::NUM_COMPONENTS};
    return detail::sort_less<Dimensionality,SIZE>()(a,(T)b);
  }
#endif



};

}
} // namespace dax::math

#endif //__dax_math_Compare_h
