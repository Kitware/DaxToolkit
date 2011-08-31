/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Types_h
#define __dax_Types_h

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

  /// \internal
  struct Vector3Struct {
    float x; float y; float z;
  } __attribute__ ((aligned(4)));

  struct Vector4Struct {
    float x; float y; float z; float w;
  } __attribute__ ((aligned(16)));

  /// \endinternal



  /// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
  /// CUDA C Programming Guide 4.0)

  /// Scalar corresponds to a single-valued floating point number.
  typedef float Scalar __attribute__ ((aligned (4)));

  /// Vector3 corresponds to a 3-tuple.
  typedef struct Vector3Struct Vector3; 

  struct Int3Struct {
    int x; int y; int z;
  } __attribute__ ((aligned(4)));
  
  /// Int3 corresponds to a integral valued, 3-tuple.
  typedef struct Int3Struct Int3;
  
  /// Vector4 corresponds to a 4-tuple.
  typedef struct Vector4Struct Vector4;

  /// Represents an ID.
  typedef int Id __attribute__ ((aligned(4)));

  /// Initializes and returns a Vector3.
  inline Vector3 make_Vector3(float x, float y, float z)
    {
    Vector3 temp;
    temp.x = x; temp.y = y; temp.z = z;
    return temp;
    }

  /// Initializes and returns a Vector4.
  inline Vector4 make_Vector4(float x, float y, float z, float w)
    {
    Vector4 temp;
    temp.x = x; temp.y = y; temp.z = z; temp.w = w;
    return temp;
    }

#else

  typedef float Scalar;
  typedef float3 Vector3;
  typedef int3 Int3;
  typedef float4 Vector4;
  typedef int Id;

  __host__ __device__ inline Vector3 make_Vector3(float x, float y, float z)
    {
    return make_float3(x, y, z);
    }

  __host__ __device__ inline Vector4 make_Vector4(float x, float y, float z, float w)
    {
    return make_float4(x, y, z, w);
    }

#endif

  inline Int3 operator+(const Int3 &a, const Int3 &b)
  {
    Int3 result = { a.x + b.x, a.y + b.y, a.z + b.z };
  }

  typedef struct StructuredPointsMetaDataStruct {
    Vector3 Origin;
    Vector3 Spacing;
    Int3 ExtentMin;
    Int3 ExtentMax;
  } StructuredPointsMetaData __attribute__ ((aligned(4)));

}
#endif
