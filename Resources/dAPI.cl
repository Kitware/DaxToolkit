/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This file provides the implementation for the dAPI.
// Any function or struct beginning with "dax" is public APi while anything
// beginning with __dax is private/internal API and can change without notice.

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
  __global __daxArrayCore* Core;
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
  __global __daxArrayCore* cores,
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
#define float3 float4

float3 daxGetArrayValue3(const daxWork* work, const daxArray* array)
{
  float3 retval;
  retval.x = retval.y = retval.z = 0.0;
  return retval;
}

void daxSetArrayValue(const daxWork* work, daxArray* output, float scalar)
{

}
