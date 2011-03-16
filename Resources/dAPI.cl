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
  __daxArrayI* Arrays;
  uchar Generator;

  float4 TempResultF;
  uint4 TempResultUI;
};
typedef struct daxArray __daxArrayI;

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

void __daxInitializeArrays(daxArray* arrays, __daxArrayCore* cores,
  const uint num_items)
{
  for (uint cc=0; cc < num_items; cc++)
    {
    arrays[cc].Core = cores[cc];
    arrays[cc].Arrays = arrays;
    arrays[cc].InputDataF = NULL;
    arrays[cc].OutputDataF = NULL;
    arrays[cc].Generator = 0;
    }
}
