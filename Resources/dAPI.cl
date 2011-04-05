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


#define printf(...)

struct __attribute__((packed)) __daxArrayCoreI
{
  uchar Type; // 0 -- irregular
              // 1 -- image-data points array
              // 2 -- image-data connections array
  uchar Rank;
  uchar Shape[2];
};

typedef struct  __attribute__((packed)) __daxArrayCoreI __daxArrayCore;

struct __daxArrayI
{
  __constant const float *InputDataF;
  __global float *OutputDataF;
  __constant const __daxArrayCore* Core;
  struct __daxArrayI* Arrays;
  uchar Generator;

  float4 TempResultF;
//  __private float4 TempResultF;
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
  __constant const __daxArrayCore* cores,
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

// PERF:  not using packed here gives a speedup of 2.
struct /*__attribute__((packed))*/ __daxImageDataData
{
  float Spacing[3] __attribute__ ((endian(host)));
  float Origin[3] __attribute__ ((endian(host)));
  unsigned int Extents[6] __attribute__ ((endian(host)));
};
typedef struct  __daxImageDataData daxImageDataData;

uint4 __daxGetDims(const daxImageDataData* imageData)
{
  uint4 dims;
  dims.x = imageData->Extents[1] - imageData->Extents[0] + 1;
  dims.y = imageData->Extents[3] - imageData->Extents[2] + 1;
  dims.z = imageData->Extents[5] - imageData->Extents[4] + 1;
  return dims;
}

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
    return retval;
    }
  else if (array->Core->Type == ARRAY_TYPE_IMAGE_POINTS)
    {
    // Now using spacing and extents and id compute the point coordinates.
    daxImageDataData imageData;
    if (array->InputDataF)
      {
      imageData = *((daxImageDataData*)(array->InputDataF));
      }
    uint4 dims = __daxGetDims(&imageData);
    // assume non-zero dims for now.
    uint4 xyz; 
    xyz.x = work->ElementID % dims.x;
    xyz.y = (work->ElementID/dims.x) % dims.y;
    xyz.z = work->ElementID / (dims.x * dims.y);
    retval.x = imageData.Origin[0] + (xyz.x + imageData.Extents[0]) * imageData.Spacing[0];
    retval.y = imageData.Origin[1] + (xyz.y + imageData.Extents[2]) * imageData.Spacing[1];
    retval.z = imageData.Origin[2] + (xyz.z + imageData.Extents[4]) * imageData.Spacing[2];
    printf("Location: %f, %f, %f\n", retval.x, retval.y, retval.z);
    return retval;
    }
  printf("__daxGetArrayValue3 case not handled %d", array->Core->Type);
  return retval;
}

float __daxGetArrayValue(const daxWork* work, const daxArray* array)
{
  printf("__daxGetArrayValue\n");
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
    else if (output->OutputDataF == 0)
      {
      // we are setting value for an intermediate array.
      // simply save the value in local memory.
      output->TempResultF.x = scalar;
      }
    }
}


// API for access topology.
struct __daxConnectedComponent
{
  uint Id;
  const daxArray* ConnectionsArray;
};

typedef struct __daxConnectedComponent daxConnectedComponent;

void daxGetConnectedComponent(const daxWork* work,
  const daxArray* connections_array,
  daxConnectedComponent* element)
{
  element->Id = work->ElementID;
  element->ConnectionsArray = connections_array;
}

uint daxGetNumberOfElements(const daxConnectedComponent* element)
{
  // For now, we assume that connections array is never generated, but a global
  // input array. Connections array will start being generated once we start
  // dealing with topology changing algorithms.
  switch (element->ConnectionsArray->Core->Type)
    {
  case ARRAY_TYPE_IMAGE_CELL:
    return 8; // assume 3D cells for now.

    }
  printf("daxGetNumberOfElements case not handled.");
  return 0;
}

void daxGetWorkForElement(const daxConnectedComponent* element,
  uint index, daxWork* element_work)
{
  printf ("daxGetWorkForElement %d\n", index);
  switch (element->ConnectionsArray->Core->Type)
    {
  case ARRAY_TYPE_IMAGE_CELL:
      {
      daxImageDataData imageData =
        *((daxImageDataData*)(element->ConnectionsArray->InputDataF));
      uint4 dims = __daxGetDims(&imageData);
      // assume non-zero dims for now.
      uint4 ijk;
      ijk.x = element->Id % (dims.x -1);
      ijk.y = (element->Id / (dims.x - 1)) % (dims.y -1);
      ijk.z = element->Id / ( (dims.x-1) * (dims.y-1) );
      ijk.x += index % 2;
      ijk.y += (index / 2) % 2;
      ijk.z += (index / 4);
      element_work->ElementID =
        ijk.x + ijk.y*dims.x + ijk.z*dims.x*dims.y;
      //printf ("cellId: %d:%d, structured id: %d %d %d, point id: %d\n",
      //  element->Id, index, loc.x, loc.y, loc.z,
      //  element_work->ElementID);
      }

    break;

  default:
    printf("daxGetWorkForElement case not handled.");
    }
}
