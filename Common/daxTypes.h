/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxTypes_h
#define __daxTypes_h

//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************

#ifndef __CUDACC__
/// Alignment requirements are prescribed by CUDA on device (Table B-1 in NVIDIA
/// CUDA C Programming Guide 4.0)
typedef float DaxScalar __attribute__ ((aligned (4)));

typedef struct DaxVector3Struct {
  float x; float y; float z;
} DaxVector3 __attribute__ ((aligned(4)));

typedef struct DaxInt3Struct {
  int x; int y; int z;
} DaxInt3 __attribute__ ((aligned(4)));

typedef struct DaxVector4Struct {
  float x; float y; float z; float w;
} DaxVector4 __attribute__ ((aligned(16)));

typedef int DaxId __attribute__ ((aligned(4)));

inline DaxVector3 make_DaxVector3(float x, float y, float z)
{
  DaxVector3 temp;
  temp.x = x; temp.y = y; temp.z = z;
  return temp;
}

inline DaxVector4 make_DaxVector4(float x, float y, float z, float w)
{
  DaxVector4 temp;
  temp.x = x; temp.y = y; temp.z = z; temp.w = w;
  return temp;
}

#else

typedef float DaxScalar;
typedef float3 DaxVector3;
typedef int3 DaxInt3;
typedef float4 DaxVector4;
typedef int DaxId;

#define make_DaxVector3 make_float3
#define make_DaxVector4 make_float4

#endif

typedef struct DaxStructuredPointsMetaDataStruct {
  DaxVector3 Origin;
  DaxVector3 Spacing;
  DaxInt3 ExtentMin;
  DaxInt3 ExtentMax;
} DaxStructuredPointsMetaData __attribute__ ((aligned(4)));

#endif
