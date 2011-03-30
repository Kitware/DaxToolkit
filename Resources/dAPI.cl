/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This file provides the implementation for the dAPI.
// Any function or struct beginning with "dax" is public APi while anything
// beginning with __dax is private/internal API and can change without notice.

#define ARRAY_TYPE_IRREGULAR 0
#define ARRAY_TYPE_IMAGE_POINTS 1
#define ARRAY_TYPE_IMAGE_CELL 2
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

struct __daxArrayCoreI
{
  uchar Type; // 0 -- irregular
              // 1 -- image-data points array
              // 2 -- image-data connections array
  uchar Rank;
  uchar Shape[2];
};

typedef struct __daxArrayCoreI __daxArrayCore;

struct __daxArrayI
{
  __global const float *InputDataF;
  __global float *OutputDataF;
  __global const __daxArrayCore* Core;
  struct __daxArrayI* Arrays;
  uchar Generator;

  float4 TempResultF;
  uint4 TempResultUI;
};
typedef struct __daxArrayI daxArray;

typedef struct
{
  uint ElementID;
} daxWork;

void __daxInitializeWorkFromGlobal(daxWork* work)
{
  // assuming 1D kernel invocations with 1-to-1 correspondence with output
  // elements and work items.
  work->ElementID = get_global_id(0);
}

void __daxInitializeArrays(daxArray* arrays,
  __global const __daxArrayCore* cores,
  const uint num_items)
{
  for (uint cc=0; cc < num_items; cc++)
    {
    arrays[cc].Core = &cores[cc];
    arrays[cc].Arrays = arrays;
    arrays[cc].InputDataF = 0;
    arrays[cc].OutputDataF = 0;
    arrays[cc].Generator = 0;
    }
}

#define __positions__
#define __and__(x,y)
#define __dep__(x)

#ifndef float3
# define float3 float4
# define dax_use_float4_for_float3
#endif

struct daxImageDataData
{
  float Spacing[3];
  float Origin[3];
  unsigned int Extents[6];
} __attribute__((packed));

float3 __daxGetArrayValue3(const daxWork* work, const daxArray* array)
{
  float3 retval =
#ifdef dax_use_float4_for_float3
    (float3)(0, 0, 0, 0)
#else
    (float3)(0,0, 0)
#endif
    ;
  if (array->Core->Type == ARRAY_TYPE_IRREGULAR)
    {
    if (array->InputDataF != 0)
      {
      // reading from global input array.
      retval.x = array->InputDataF[3*work->ElementID];
      retval.y = array->InputDataF[3*work->ElementID + 1];
      retval.z = array->InputDataF[3*work->ElementID + 2];
      }
    else 
      {
      // reading from temporary array.
      retval.x = array->TempResultF.x;
      retval.y = array->TempResultF.y;
      retval.z = array->TempResultF.z;
      }
    }
  return retval;
}

float __daxGetArrayValue(const daxWork* work, const daxArray* array)
{
  if (array->Core->Type == ARRAY_TYPE_IRREGULAR)
    {
    if (array->InputDataF != 0)
      {
      // reading from global input array.
      return array->InputDataF[work->ElementID];
      }
    else 
      {
      // reading from temporary array.
      return array->TempResultF.x;
      }
    }

  return 0.0;
}

void daxSetArrayValue(const daxWork* work, daxArray* output, float scalar)
{
  if (output->Core->Type == ARRAY_TYPE_IRREGULAR)
    {
    if (output->OutputDataF != 0)
      {
      // indicates we are writing to a global out-array.
      //printf("%d == %f, \t",work->ElementID, scalar);
      output->OutputDataF[work->ElementID] = scalar;
      }
    else if (output->InputDataF == 0)
      {
      // we are setting value for an intermediate array.
      // simply save the value in local memory.
      output->TempResultF.x = scalar;
      }
    }
}
