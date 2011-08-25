/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Types_h
#define __dax_Types_h

/*! \namespace dax
 *  dax is the namespace for the Dax Toolkit. It contains other sub namespaces,
 *  as well as basic data types and functions callable from all components in
 *  Dax toolkit.
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

  typedef struct StructuredPointsMetaDataStruct {
    Vector3 Origin;
    Vector3 Spacing;
    Int3 ExtentMin;
    Int3 ExtentMax;
  } StructuredPointsMetaData __attribute__ ((aligned(4)));

}
#endif
