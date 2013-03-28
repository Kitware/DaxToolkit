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
#ifndef __dax_Types_h
#define __dax_Types_h

#include <dax/internal/Configure.h>
#include <dax/internal/ExportMacros.h>

/*!
 * \namespace dax
 * \brief Dax Toolkit.
 *
 * dax is the namespace for the Dax Toolkit. It contains other sub namespaces,
 * as well as basic data types and functions callable from all components in Dax
 * toolkit.
 *
 * \namespace dax::cont
 * \brief Dax Control Environment.
 *
 * dax::cont defines the publicly accessible API for the Dax Control
 * Environment. Users of the Dax Toolkit can use this namespace to access the
 * Control Environment.
 *
 * \namespace dax::cuda
 * \brief CUDA implementation.
 *
 * dax::cuda includes the code to implement the Dax for CUDA-based platforms.
 *
 * \namespace dax::cuda::cont
 * \brief CUDA implementation for Control Environment.
 *
 * dax::cuda::cont includes the code to implement the Dax Control Environment
 * for CUDA-based platforms.
 *
 * \namespace dax::cuda::exec
 * \brief CUDA implementation for Execution Environment.
 *
 * dax::cuda::exec includes the code to implement the Dax Execution Environment
 * for CUDA-based platforms.
 *
 * \namespace dax::exec
 * \brief Dax Execution Environment.
 *
 * dax::exec defines the publicly accessible API for the Dax Execution
 * Environment. Worklets typically use classes/apis defined within this
 * namespace alone.
 *
 * \namespace dax::internal
 * \brief Dax Internal Environment
 *
 * dax::internal defines API which is internal and subject to frequent
 * change. This should not be used for projects using Dax. Instead it servers
 * are a reference for the developers of Dax.
 *
 * \namespace dax::math
 * \brief Utility math functions
 *
 * dax::math defines the publicly accessible API for Utility Math functions.
 *
 * \namespace dax::testing
 * \brief Internal testing classes
 *
 */

namespace dax
{
//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************

/// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
/// CUDA C Programming Guide 4.0)

namespace internal {

#if DAX_SIZE_INT == 4
typedef int Int32Type;
typedef unsigned int UInt32Type;
#else
#error Could not find a 32-bit integer.
#endif

#if DAX_SIZE_LONG == 8
typedef long Int64Type;
typedef unsigned long UInt64Type;
#elif DAX_SIZE_LONG_LONG == 8
typedef long long Int64Type;
typedef unsigned long long UInt64Type;
#else
#error Could not find a 64-bit integer.
#endif

//-----------------------------------------------------------------------------

template<int Size>
struct equals
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    return equals<Size-1>()(a,b) && a[Size-1] == b[Size-1];
  }
};

template<>
struct equals<1>
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    return a[0] == b[0];
  }
};

template<>
struct equals<2>
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    return a[0] == b[0] && a[1] == b[1];
  }
};

template<>
struct equals<3>
{
  template<typename T>
  DAX_EXEC_CONT_EXPORT bool operator()(const T& a, const T& b) const
  {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
  }
};

template<int Size>
struct assign_scalar_to_vector
{
  template<typename VectorType, typename ComponentType>
  DAX_EXEC_CONT_EXPORT
  void operator()(VectorType &dest, const ComponentType &src)
  {
    assign_scalar_to_vector<Size-1>()(dest, src);
    dest[Size-1] = src;
  }
};

template<>
struct assign_scalar_to_vector<1>
{
  template<typename VectorType, typename ComponentType>
  DAX_EXEC_CONT_EXPORT
  void operator()(VectorType &dest, const ComponentType &src)
  {
    dest[0] = src;
  }
};

template<>
struct assign_scalar_to_vector<2>
{
  template<typename VectorType, typename ComponentType>
  DAX_EXEC_CONT_EXPORT
  void operator()(VectorType &dest, const ComponentType &src)
  {
    dest[0] = src; dest[1] = src;
  }
};

template<>
struct assign_scalar_to_vector<3>
{
  template<typename VectorType, typename ComponentType>
  DAX_EXEC_CONT_EXPORT
  void operator()(VectorType &dest, const ComponentType &src)
  {
    dest[0] = src; dest[1] = src; dest[2] = src;
  }
};

template<int Size>
struct copy_vector
{
  template<typename T1, typename T2>
  DAX_EXEC_CONT_EXPORT void operator()(T1 &dest, const T2 &src)
  {
    copy_vector<Size-1>()(dest, src);
    dest[Size-1] = src[Size-1];
  }
};

template<>
struct copy_vector<1>
{
  template<typename T1, typename T2>
  DAX_EXEC_CONT_EXPORT void operator()(T1 &dest, const T2 &src)
  {
    dest[0] = src[0];
  }
};

template<>
struct copy_vector<2>
{
  template<typename T1, typename T2>
  DAX_EXEC_CONT_EXPORT void operator()(T1 &dest, const T2 &src)
  {
    dest[0] = src[0]; dest[1] = src[1];
  }
};

template<>
struct copy_vector<3>
{
  template<typename T1, typename T2>
  DAX_EXEC_CONT_EXPORT void operator()(T1 &dest, const T2 &src)
  {
    dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2];
  }
};

} // namespace internal

//-----------------------------------------------------------------------------

#if DAX_SIZE_ID == 4

/// Represents an ID.
typedef internal::Int32Type Id __attribute__ ((aligned(DAX_SIZE_ID)));

#elif DAX_SIZE_ID == 8

/// Represents an ID.
typedef internal::Int64Type Id __attribute__ ((aligned(DAX_SIZE_ID)));

#else
#error Unknown Id Size
#endif

#ifdef DAX_USE_DOUBLE_PRECISION

/// Scalar corresponds to a floating point number.
typedef double Scalar __attribute__ ((aligned(DAX_SIZE_SCALAR)));

#else //DAX_USE_DOUBLE_PRECISION

/// Scalar corresponds to a floating point number.
typedef float Scalar __attribute__ ((aligned(DAX_SIZE_SCALAR)));

#endif //DAX_USE_DOUBLE_PRECISION

//-----------------------------------------------------------------------------

/// Tuple corresponds to a Size-tuple of type T
template<typename T, int Size>
class Tuple {
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS=Size;

  DAX_EXEC_CONT_EXPORT Tuple(){}
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType& value)
    {
    for(int i=0; i < NUM_COMPONENTS;++i)
      {
      this->Components[i]=value;
      }
    }
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType* values)
    {
    for(int i=0; i < NUM_COMPONENTS;++i)
      {
      this->Components[i]=values[i];
      }
    }
  DAX_EXEC_CONT_EXPORT
  Tuple(const Tuple<ComponentType, Size> &src)
  {
    for (int i = 0; i < NUM_COMPONENTS; i++)
      {
      this->Components[i] = src[i];
      }
  }

  DAX_EXEC_CONT_EXPORT
  Tuple<ComponentType, Size> &operator=(const Tuple<ComponentType, Size> &src)
  {
    for (int i = 0; i < NUM_COMPONENTS; i++)
      {
      this->Components[i] = src[i];
      }
    return *this;
  }

  DAX_EXEC_CONT_EXPORT const ComponentType &operator[](int idx) const {
    return this->Components[idx];
  }
  DAX_EXEC_CONT_EXPORT ComponentType &operator[](int idx) {
    return this->Components[idx];
  }

  DAX_EXEC_CONT_EXPORT
  bool operator==(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    bool same = true;
    for (int componentIndex=0; componentIndex<NUM_COMPONENTS; componentIndex++)
      {
      same &= (this->Components[componentIndex] == other[componentIndex]);
      }
    return same;
  }
  DAX_EXEC_CONT_EXPORT
  bool operator!=(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    bool same = true;
    for (int componentIndex=0; componentIndex<NUM_COMPONENTS; componentIndex++)
      {
      same &= (this->Components[componentIndex] != other[componentIndex]);
      }
    return same;
  }

protected:
  ComponentType Components[NUM_COMPONENTS];
};

//-----------------------------------------------------------------------------
// Specializations for common tuple sizes (with special names).

template<typename T>
class Tuple<T,2>{
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS = 2;

  DAX_EXEC_CONT_EXPORT Tuple(){}
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType& value)
  {
    internal::assign_scalar_to_vector<NUM_COMPONENTS>()(this->Components,value);
  }
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType* values)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, values);
  }
  DAX_EXEC_CONT_EXPORT Tuple(ComponentType x, ComponentType y) {
    this->Components[0] = x;
    this->Components[1] = y;
  }
  DAX_EXEC_CONT_EXPORT
  Tuple(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
  }

  DAX_EXEC_CONT_EXPORT
  Tuple<ComponentType, NUM_COMPONENTS> &
  operator=(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
    return *this;
  }

  DAX_EXEC_CONT_EXPORT const ComponentType &operator[](int idx) const {
    return this->Components[idx];
  }
  DAX_EXEC_CONT_EXPORT ComponentType &operator[](int idx) {
    return this->Components[idx];
  }

  DAX_EXEC_CONT_EXPORT
  bool operator==(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return internal::equals<NUM_COMPONENTS>()(*this, other);
  }
  DAX_EXEC_CONT_EXPORT
  bool operator!=(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return !(this->operator==(other));
  }

protected:
  ComponentType Components[NUM_COMPONENTS];
};

/// Vector2 corresponds to a 2-tuple
typedef dax::Tuple<dax::Scalar,2>
    Vector2 __attribute__ ((aligned(DAX_SIZE_TWO_SCALAR)));

template<typename T>
class Tuple<T,3>{
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS = 3;

  DAX_EXEC_CONT_EXPORT Tuple(){}
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType& value)
  {
    internal::assign_scalar_to_vector<NUM_COMPONENTS>()(this->Components,value);
  }
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType* values)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, values);
  }
  DAX_EXEC_CONT_EXPORT
  Tuple(ComponentType x, ComponentType y, ComponentType z) {
    this->Components[0] = x;
    this->Components[1] = y;
    this->Components[2] = z;
  }
  DAX_EXEC_CONT_EXPORT
  Tuple(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
  }

  DAX_EXEC_CONT_EXPORT
  Tuple<ComponentType, NUM_COMPONENTS> &
  operator=(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
    return *this;
  }

  DAX_EXEC_CONT_EXPORT const ComponentType &operator[](int idx) const {
    return this->Components[idx];
  }
  DAX_EXEC_CONT_EXPORT ComponentType &operator[](int idx) {
    return this->Components[idx];
  }

  DAX_EXEC_CONT_EXPORT
  bool operator==(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return internal::equals<NUM_COMPONENTS>()(*this, other);
  }
  DAX_EXEC_CONT_EXPORT
  bool operator!=(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return !(this->operator==(other));
  }

protected:
  ComponentType Components[NUM_COMPONENTS];
};

/// Vector3 corresponds to a 3-tuple
typedef dax::Tuple<dax::Scalar,3>
    Vector3 __attribute__ ((aligned(DAX_SIZE_SCALAR)));

template<typename T>
class Tuple<T,4>{
public:
  typedef T ComponentType;
  static const int NUM_COMPONENTS = 4;

  DAX_EXEC_CONT_EXPORT Tuple(){}
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType& value)
  {
    internal::assign_scalar_to_vector<NUM_COMPONENTS>()(this->Components,value);
  }
  DAX_EXEC_CONT_EXPORT explicit Tuple(const ComponentType* values)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, values);
  }
  DAX_EXEC_CONT_EXPORT
  Tuple(ComponentType x, ComponentType y, ComponentType z, ComponentType w) {
    this->Components[0] = x;
    this->Components[1] = y;
    this->Components[2] = z;
    this->Components[3] = w;
  }
  DAX_EXEC_CONT_EXPORT
  Tuple(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
  }

  DAX_EXEC_CONT_EXPORT
  Tuple<ComponentType, NUM_COMPONENTS> &
  operator=(const Tuple<ComponentType, NUM_COMPONENTS> &src)
  {
    internal::copy_vector<NUM_COMPONENTS>()(this->Components, src.Components);
    return *this;
  }

  DAX_EXEC_CONT_EXPORT const ComponentType &operator[](int idx) const {
    return this->Components[idx];
  }
  DAX_EXEC_CONT_EXPORT ComponentType &operator[](int idx) {
    return this->Components[idx];
  }

  DAX_EXEC_CONT_EXPORT
  bool operator==(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return internal::equals<NUM_COMPONENTS>()(*this, other);
  }
  DAX_EXEC_CONT_EXPORT
  bool operator!=(const Tuple<T,NUM_COMPONENTS> &other) const
  {
    return !(this->operator==(other));
  }

protected:
  ComponentType Components[NUM_COMPONENTS];
};

/// Vector4 corresponds to a 4-tuple
typedef dax::Tuple<dax::Scalar,4>
    Vector4 __attribute__ ((aligned(DAX_SIZE_FOUR_SCALAR)));


/// Id3 corresponds to a 3-dimensional index for 3d arrays.  Note that
/// the precision of each index may be less than dax::Id.
typedef dax::Tuple<dax::Id,3> Id3 __attribute__ ((aligned(DAX_SIZE_ID)));

/// Initializes and returns a Vector2.
DAX_EXEC_CONT_EXPORT dax::Vector2 make_Vector2(dax::Scalar x,
                                               dax::Scalar y)
{
  return dax::Vector2(x, y);
}

/// Initializes and returns a Vector3.
DAX_EXEC_CONT_EXPORT dax::Vector3 make_Vector3(dax::Scalar x,
                                               dax::Scalar y,
                                               dax::Scalar z)
{
  return dax::Vector3(x, y, z);
}

/// Initializes and returns a Vector4.
DAX_EXEC_CONT_EXPORT dax::Vector4 make_Vector4(dax::Scalar x,
                                               dax::Scalar y,
                                               dax::Scalar z,
                                               dax::Scalar w)
{
  return dax::Vector4(x, y, z, w);
}

/// Initializes and returns an Id3
DAX_EXEC_CONT_EXPORT dax::Id3 make_Id3(dax::Id x, dax::Id y, dax::Id z)
{
  return dax::Id3(x, y, z);
}

template<typename T, int Size>
DAX_EXEC_CONT_EXPORT T dot(const dax::Tuple<T,Size> &a,
                           const dax::Tuple<T,Size> &b)
{
  T result = a[0]*b[0];
  for (int componentIndex = 1; componentIndex < Size; componentIndex++)
    {
    result += a[componentIndex]*b[componentIndex];
    }
  return result;
}

DAX_EXEC_CONT_EXPORT dax::Id dot(dax::Id a, dax::Id b)
{
  return a * b;
}

DAX_EXEC_CONT_EXPORT dax::Scalar dot(dax::Scalar a, dax::Scalar b)
{
  return a * b;
}

} // End of namespace dax

template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator+(const dax::Tuple<T,Size> &a,
                                                  const dax::Tuple<T,Size> &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] + b[componentIndex];
    }
  return result;
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator-(const dax::Tuple<T,Size> &a,
                                                  const dax::Tuple<T,Size> &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] - b[componentIndex];
    }
  return result;
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator*(const dax::Tuple<T,Size> &a,
                                                  const dax::Tuple<T,Size> &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] * b[componentIndex];
    }
  return result;
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator/(const dax::Tuple<T,Size> &a,
                                                  const dax::Tuple<T,Size> &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] / b[componentIndex];
    }
  return result;
}

template<typename Ta, typename Tb, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<Ta,Size> operator*(const dax::Tuple<Ta,Size> &a,
                                                   const Tb &b)
{
  dax::Tuple<Ta,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] * b;
    }
  return result;
}
template<typename Ta, typename Tb, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<Tb,Size> operator*(const Ta &a,
                                                   const dax::Tuple<Tb,Size> &b)
{
  dax::Tuple<Tb,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a * b[componentIndex];
    }
  return result;
}

#endif //__dax_Types_h
