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
 * as well as basic data types and functions callable from all components in
 * Dax toolkit.
 *
 * \namespace dax::cont
 * \brief Dax Control Environment.
 *
 * dax::cont defines the publicly accessible API for the Dax Control
 * Environment. Users of the Dax Toolkit can use this namespace to access the
 * Control Environment.
 *
 * \namespace dax::exec
 * \brief Dax Execution Environment.
 *
 * dax::exec defines the publicly accessible API for the Dax Execution
 * Environment. Worklets typically use classes/apis defined within this
 * namespace alone.
 *
 * \namespace dax::cuda
 * \brief CUDA implementation.
 *
 * dax::cuda includes the code to implement the Dax for CUDA-based
 * platforms.
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

} // namespace internal

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

  DAX_EXEC_CONT_EXPORT const ComponentType &operator[](int idx) const {
    return this->Components[idx];
  }
  DAX_EXEC_CONT_EXPORT ComponentType &operator[](int idx) {
    return this->Components[idx];
  }

protected:
  ComponentType Components[NUM_COMPONENTS];
};

/// Vector2 corresponds to a 2-tuple
class Vector2 : public dax::Tuple<dax::Scalar,2>{
public:

  DAX_EXEC_CONT_EXPORT Vector2() {}
  DAX_EXEC_CONT_EXPORT explicit Vector2(const dax::Scalar& value):
    dax::Tuple<dax::Scalar, 2>(value){}
  DAX_EXEC_CONT_EXPORT explicit Vector2(const dax::Scalar* values):
    dax::Tuple<dax::Scalar, 2>(values) { }
  DAX_EXEC_CONT_EXPORT Vector2(const dax::Tuple<dax::Scalar,2> &values)
    : dax::Tuple<dax::Scalar, 2>(values) { }

  DAX_EXEC_CONT_EXPORT Vector2(ComponentType x, ComponentType y) {
    this->Components[0] = x;
    this->Components[1] = y;
  }
} __attribute__ ((aligned(DAX_SIZE_SCALAR)));

/// Vector3 corresponds to a 3-tuple
class Vector3 : public dax::Tuple<dax::Scalar,3>{
public:

  DAX_EXEC_CONT_EXPORT Vector3() {}
  DAX_EXEC_CONT_EXPORT explicit Vector3(const dax::Scalar& value):
    dax::Tuple<dax::Scalar, 3>(value){}
  DAX_EXEC_CONT_EXPORT explicit Vector3(const dax::Scalar* values):
    dax::Tuple<dax::Scalar, 3>(values) { }
  DAX_EXEC_CONT_EXPORT Vector3(const dax::Tuple<dax::Scalar,3> &values)
    : dax::Tuple<dax::Scalar, 3>(values) { }

  DAX_EXEC_CONT_EXPORT
  Vector3(ComponentType x, ComponentType y, ComponentType z) {
    this->Components[0] = x;
    this->Components[1] = y;
    this->Components[2] = z;
  }
} __attribute__ ((aligned(DAX_SIZE_SCALAR)));

/// Vector4 corresponds to a 4-tuple
class Vector4 : public dax::Tuple<Scalar,4>{
public:

  DAX_EXEC_CONT_EXPORT Vector4() {}
  DAX_EXEC_CONT_EXPORT explicit Vector4(const dax::Scalar& value):
    dax::Tuple<dax::Scalar, 4>(value){}
  DAX_EXEC_CONT_EXPORT explicit Vector4(const dax::Scalar* values):
    dax::Tuple<dax::Scalar, 4>(values) { }
  DAX_EXEC_CONT_EXPORT Vector4(const dax::Tuple<dax::Scalar,4> &values)
    : dax::Tuple<dax::Scalar, 4>(values) { }

  DAX_EXEC_CONT_EXPORT
  Vector4(ComponentType x, ComponentType y, ComponentType z, ComponentType w) {
    this->Components[0] = x;
    this->Components[1] = y;
    this->Components[2] = z;
    this->Components[3] = w;
  }
} __attribute__ ((aligned(DAX_SIZE_SCALAR)));


/// Id3 corresponds to a 3-dimensional index for 3d arrays.  Note that
/// the precision of each index may be less than dax::Id.
class Id3 : public dax::Tuple<dax::Id,3>{
public:

  DAX_EXEC_CONT_EXPORT Id3() {}
  DAX_EXEC_CONT_EXPORT explicit Id3(const dax::Id& value):
    dax::Tuple<dax::Id, 3>(value){}
  DAX_EXEC_CONT_EXPORT explicit Id3(const dax::Id* values):
    dax::Tuple<dax::Id, 3>(values) { }
  DAX_EXEC_CONT_EXPORT Id3(const dax::Tuple<dax::Id,3> &values)
    : dax::Tuple<dax::Id, 3>(values) { }

  DAX_EXEC_CONT_EXPORT Id3(ComponentType x, ComponentType y, ComponentType z) {
    this->Components[0] = x;
    this->Components[1] = y;
    this->Components[2] = z;
  }
} __attribute__ ((aligned(DAX_SIZE_ID)));

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

DAX_EXEC_CONT_EXPORT dax::Id3::ComponentType dot(const dax::Id3 &a,
                                                 const dax::Id3 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]);
}

DAX_EXEC_CONT_EXPORT dax::Scalar dot(dax::Scalar a, dax::Scalar b)
{
  return a * b;
}

DAX_EXEC_CONT_EXPORT dax::Vector2::ComponentType dot(const dax::Vector2 &a,
                                                 const dax::Vector2 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]);
}

DAX_EXEC_CONT_EXPORT dax::Vector3::ComponentType dot(const dax::Vector3 &a,
                                                     const dax::Vector3 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]);
}

DAX_EXEC_CONT_EXPORT dax::Vector4::ComponentType dot(const dax::Vector4 &a,
                                                     const dax::Vector4 &b)
{
  return (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]) + (a[3]*b[3]);
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
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Tuple<T,Size> &a,
                                     const dax::Tuple<T,Size> &b)
{
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    if (a[componentIndex] != b[componentIndex]) return false;
    }
  return true;
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Tuple<T,Size> &a,
                                     const dax::Tuple<T,Size> &b)
{
  return !(a == b);
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator*(const dax::Tuple<T,Size> &a,
                                                  const T &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a[componentIndex] * b;
    }
  return result;
}
template<typename T, int Size>
DAX_EXEC_CONT_EXPORT dax::Tuple<T,Size> operator*(const T &a,
                                                  const dax::Tuple<T,Size> &b)
{
  dax::Tuple<T,Size> result;
  for (int componentIndex = 0; componentIndex < Size; componentIndex++)
    {
    result[componentIndex] = a * b[componentIndex];
    }
  return result;
}

DAX_EXEC_CONT_EXPORT dax::Id3 operator+(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator*(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator-(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator/(const dax::Id3 &a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Id3 &a,
                                     const dax::Id3 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Id3 &a,
                                     const dax::Id3 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Id3 operator*(dax::Id3::ComponentType a,
                                        const dax::Id3 &b)
{
  return dax::make_Id3(a*b[0], a*b[1], a*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Id3 operator*(const dax::Id3 &a,
                                        dax::Id3::ComponentType &b)
{
  return dax::make_Id3(a[0]*b, a[1]*b, a[2]*b);
}

DAX_EXEC_CONT_EXPORT dax::Vector2 operator+(const dax::Vector2 &a,
                                            const dax::Vector2 &b)
{
  return dax::make_Vector2(a[0]+b[0], a[1]+b[1]);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 operator*(const dax::Vector2 &a,
                                            const dax::Vector2 &b)
{
  return dax::make_Vector2(a[0]*b[0], a[1]*b[1]);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 operator-(const dax::Vector2 &a,
                                            const dax::Vector2 &b)
{
  return dax::make_Vector2(a[0]-b[0], a[1]-b[1]);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 operator/(const dax::Vector2 &a,
                                            const dax::Vector2 &b)
{
  return dax::make_Vector2(a[0]/b[0], a[1]/b[1]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Vector2 &a,
                                     const dax::Vector2 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Vector2 &a,
                                     const dax::Vector2 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Vector2 operator*(dax::Vector2::ComponentType a,
                                            const dax::Vector2 &b)
{
  return dax::make_Vector2(a*b[0], a*b[1]);
}
DAX_EXEC_CONT_EXPORT dax::Vector2 operator*(const dax::Vector2 &a,
                                            dax::Vector2::ComponentType &b)
{
  return dax::make_Vector2(a[0]*b, a[1]*b);
}

DAX_EXEC_CONT_EXPORT dax::Vector3 operator+(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator-(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator/(const dax::Vector3 &a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Vector3 &a,
                                     const dax::Vector3 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Vector3 &a,
                                     const dax::Vector3 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(dax::Vector3::ComponentType a,
                                            const dax::Vector3 &b)
{
  return dax::make_Vector3(a*b[0], a*b[1], a*b[2]);
}
DAX_EXEC_CONT_EXPORT dax::Vector3 operator*(const dax::Vector3 &a,
                                            dax::Vector3::ComponentType &b)
{
  return dax::make_Vector3(a[0]*b, a[1]*b, a[2]*b);
}

DAX_EXEC_CONT_EXPORT dax::Vector4 operator+(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator-(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator/(const dax::Vector4 &a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a[0]/b[0], a[1]/b[1], a[2]/b[2], a[3]/b[3]);
}
DAX_EXEC_CONT_EXPORT bool operator==(const dax::Vector4 &a,
                                     const dax::Vector4 &b)
{
  return (a[0] == b[0]) && (a[1] == b[1]) && (a[2] == b[2]) && (a[3] == b[3]);
}
DAX_EXEC_CONT_EXPORT bool operator!=(const dax::Vector4 &a,
                                     const dax::Vector4 &b)
{
  return !(a == b);
}

DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(dax::Vector4::ComponentType a,
                                            const dax::Vector4 &b)
{
  return dax::make_Vector4(a*b[0], a*b[1], a*b[2], a*b[3]);
}
DAX_EXEC_CONT_EXPORT dax::Vector4 operator*(const dax::Vector4 &a,
                                            dax::Scalar &b)
{
  return dax::make_Vector4(a[0]*b, a[1]*b, a[2]*b, a[3]*b);
}

namespace dax
{
DAX_EXEC_CONT_EXPORT dax::Vector3 cross(const dax::Vector3 &a,
                                        const dax::Vector3 &b)
{
  return dax::make_Vector3 (a[1]*b[2] - a[2]*b[1],
                            a[2]*b[0] - a[0]*b[2],
                            a[0]*b[1] - a[1]*b[0]);
}

DAX_EXEC_CONT_EXPORT dax::Vector3 normal(const dax::Vector3 &a,
                                         const dax::Vector3 &b,
                                         const dax::Vector3 &c)
{
  return dax::cross ( c-b, a-b );
}
}
#endif
