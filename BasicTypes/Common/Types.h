/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Types_h
#define __dax_Types_h

namespace dax
{
  //*****************************************************************************
  // Typedefs for basic types.
  //*****************************************************************************

#ifndef __CUDACC__
  /// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
  /// CUDA C Programming Guide 4.0)
  typedef float Scalar __attribute__ ((aligned (4)));

  typedef struct Vector3Struct {
    float x; float y; float z;
  } Vector3 __attribute__ ((aligned(4)));

  typedef struct Int3Struct {
    int x; int y; int z;
  } Int3 __attribute__ ((aligned(4)));

  typedef struct Vector4Struct {
    float x; float y; float z; float w;
  } Vector4 __attribute__ ((aligned(16)));

  typedef int Id __attribute__ ((aligned(4)));

  inline Vector3 make_Vector3(float x, float y, float z)
    {
    Vector3 temp;
    temp.x = x; temp.y = y; temp.z = z;
    return temp;
    }

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
