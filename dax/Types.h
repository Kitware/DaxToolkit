/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Types_h
#define __dax_Types_h

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

#ifndef __CUDACC__

  /// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
  /// CUDA C Programming Guide 4.0)

  /// Scalar corresponds to a single-valued floating point number.
  typedef float Scalar __attribute__ ((aligned (4)));

  /// Vector3 corresponds to a 3-tuple
  struct Vector3 {
    Scalar x; Scalar y; Scalar z;
  } __attribute__ ((aligned(4)));

  /// Vector4 corresponds to a 4-tuple
  struct Vector4 {
    Scalar x; Scalar y; Scalar z; Scalar w;
  } __attribute__ ((aligned(16)));

  /// Id3 corresponds to a 3-dimensional index for 3d arrays.  Note that
  /// the precision of each index may be less than dax::Id.
  struct Id3 {
    int x; int y; int z;
  } __attribute__ ((aligned(4)));

  /// Represents an ID.
  typedef int Id __attribute__ ((aligned(4)));

  /// Initializes and returns a Vector3.
  DAX_EXEC_CONT_EXPORT inline Vector3 make_Vector3(Scalar x, Scalar y, Scalar z)
    {
    Vector3 temp;
    temp.x = x; temp.y = y; temp.z = z;
    return temp;
    }

  /// Initializes and returns a Vector4.
  DAX_EXEC_CONT_EXPORT inline
  Vector4 make_Vector4(Scalar x, Scalar y, Scalar z, Scalar w)
    {
    Vector4 temp;
    temp.x = x; temp.y = y; temp.z = z; temp.w = w;
    return temp;
    }

  /// Initializes and returns an Id3
  DAX_EXEC_CONT_EXPORT inline Id3 make_Id3(int x, int y, int z)
    {
    Id3 temp;
    temp.x = x;  temp.y = y;  temp.z = z;
    return temp;
    }

#else

  typedef float Scalar;
  typedef float3 Vector3;
  typedef int3 Id3;
  typedef float4 Vector4;
  typedef int Id;

  DAX_EXEC_CONT_EXPORT inline Vector3 make_Vector3(float x, float y, float z)
    {
    return make_float3(x, y, z);
    }

  DAX_EXEC_CONT_EXPORT inline
  Vector4 make_Vector4(float x, float y, float z, float w)
    {
    return make_float4(x, y, z, w);
    }

  DAX_EXEC_CONT_EXPORT inline Id3 make_Id3(int x, int y, int z)
    {
    return make_int3(x, y, z);
    }

#endif

  DAX_EXEC_CONT_EXPORT inline Id3 operator+(const Id3 &a, const Id3 &b)
  {
    Id3 result = { a.x + b.x, a.y + b.y, a.z + b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Id3 operator*(const Id3 &a, const Id3 &b)
  {
    Id3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Id3 operator-(const Id3 &a, const Id3 &b)
  {
    Id3 result = { a.x - b.x, a.y - b.y, a.z - b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Id3 operator/(const Id3 &a, const Id3 &b)
  {
    Id3 result = { a.x / b.x, a.y / b.y, a.z / b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline bool operator==(const Id3 &a, const Id3 &b)
  {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
  }
  DAX_EXEC_CONT_EXPORT inline bool operator!=(const Id3 &a, const Id3 &b)
  {
    return !(a == b);
  }

  DAX_EXEC_CONT_EXPORT inline Vector3
  operator+(const Vector3 &a, const Vector3 &b)
  {
    Vector3 result = { a.x + b.x, a.y + b.y, a.z + b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Vector3
  operator*(const Vector3 &a, const Vector3 &b)
  {
    Vector3 result = { a.x * b.x, a.y * b.y, a.z * b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Vector3
  operator-(const Vector3 &a, const Vector3 &b)
  {
    Vector3 result = { a.x - b.x, a.y - b.y, a.z - b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Vector3
  operator/(const Vector3 &a, const Vector3 &b)
  {
    Vector3 result = { a.x / b.x, a.y / b.y, a.z / b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline bool operator==(const Vector3 &a, const Vector3 &b)
  {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
  }
  DAX_EXEC_CONT_EXPORT inline bool operator!=(const Vector3 &a, const Vector3 &b)
  {
    return !(a == b);
  }

  DAX_EXEC_CONT_EXPORT inline Vector3 operator*(Scalar a, const Vector3 &b)
  {
    Vector3 result = { a * b.x, a * b.y, a * b.z };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Vector3 operator*(const Vector3 &a, Scalar &b)
  {
    Vector3 result = { a.x * b, a.y * b, a.z * b };
    return result;
  }

  DAX_EXEC_CONT_EXPORT inline
  Vector4 operator+(const Vector4 &a, const Vector4 &b)
  {
    Vector4 result = { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline
  Vector4 operator*(const Vector4 &a, const Vector4 &b)
  {
    Vector4 result = { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline
  Vector4 operator-(const Vector4 &a, const Vector4 &b)
  {
    Vector4 result = { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline
  Vector4 operator/(const Vector4 &a, const Vector4 &b)
  {
    Vector4 result = { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline bool operator==(const Vector4 &a, const Vector4 &b)
  {
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
  }
  DAX_EXEC_CONT_EXPORT inline bool operator!=(const Vector4 &a, const Vector4 &b)
  {
    return !(a == b);
  }

  DAX_EXEC_CONT_EXPORT inline Vector4 operator*(Scalar a, const Vector4 &b)
  {
    Vector4 result = { a * b.x, a * b.y, a * b.z, a * b.w };
    return result;
  }
  DAX_EXEC_CONT_EXPORT inline Vector4 operator*(const Vector4 &a, Scalar &b)
  {
    Vector4 result = { a.x * b, a.y * b, a.z * b, a.w * b };
    return result;
  }

}
#endif
