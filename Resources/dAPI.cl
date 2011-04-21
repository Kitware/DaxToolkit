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
#define ARRAY_TYPE_IMAGE_LINK 3
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable

#ifndef float3
# define dax_use_float4_for_float3
#endif

// Define primitive data types.
typedef float4 daxFloat4;
typedef float daxFloat;
typedef uint daxUInt;
typedef uint4 daxUInt4;
typedef uint daxIdType;
#ifdef dax_use_float4_for_float3
  typedef float4 daxFloat3;
  typedef uint4 daxUInt3;
#else
  typedef float3 daxFloat3;
  typedef uint3 daxUInt3;
#endif

// initializer functions
daxFloat3 as_daxFloat3(const daxFloat x, const daxFloat y, const daxFloat z)
{
  daxFloat3 ret;
  ret.x = x; ret.y = y; ret.z = z;
#ifdef dax_use_float4_for_float3
  ret.w = 1;
#endif
  return ret;
}

daxFloat4 as_daxFloat4(
  const daxFloat x, const daxFloat y, const daxFloat z, const daxFloat w)
{
  daxFloat4 ret;
  ret.x = x; ret.y = y; ret.z = z; ret.w = w;
  return ret;
}

daxUInt3 as_daxUInt3(const daxUInt x, const daxUInt y, const daxUInt z)
{
  daxUInt3 ret;
  ret.x = x; ret.y = y; ret.z = z;
#ifdef dax_use_float4_for_float3
  ret.w = 1;
#endif
  return ret;
}

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

  daxFloat4 TempResultF;
};
typedef struct __daxArrayI daxArray;

typedef struct
{
  daxIdType ElementID;
} daxWork;

void __daxInitializeWorkFromGlobal(daxWork* work)
{
  // assuming 1D kernel invocations with 1-to-1 correspondence with output
  // elements and work items.
  work->ElementID = get_global_id(0);
}

void __daxInitializeArrays(daxArray* arrays,
  __constant const __daxArrayCore* cores,
  const daxIdType num_items)
{
  for (daxIdType cc=0; cc < num_items; cc++)
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
#define __connections__

struct __attribute__((packed)) __daxImageDataData
{
  daxFloat Spacing[3] __attribute__ ((endian(host)));
  daxFloat Origin[3] __attribute__ ((endian(host)));
  daxUInt Extents[6] __attribute__ ((endian(host)));
};
typedef struct  __daxImageDataData daxImageDataData;

daxUInt3 __daxGetDims(const daxImageDataData* imageData)
{
  daxUInt3 dims;
  dims.x = imageData->Extents[1] - imageData->Extents[0] + 1;
  dims.y = imageData->Extents[3] - imageData->Extents[2] + 1;
  dims.z = imageData->Extents[5] - imageData->Extents[4] + 1;
  return dims;
}

daxUInt3 __daxGetPointIJK(daxUInt pid, daxUInt3 dims)
{
  daxUInt3 point_loc;
  point_loc.x = pid % dims.x;
  point_loc.y = (pid / dims.x) % dims.y;
  point_loc.z = pid / (dims.x * dims.y);
  return point_loc;
}

daxUInt3 __daxGetCellIJK(daxUInt cid, daxUInt3 dims)
{
  dims.xyz -= as_daxUInt3(1, 1, 1).xyz;
  return __daxGetPointIJK(cid, dims);
}

// returns the number of cells the point in contained in, and updates cell_ids
// to match the cells.
daxUInt __daxGetStructuredPointCells(daxIdType pid,
  daxUInt3 dims,
  daxUInt cell_ids[8])
{
  daxUInt3 upoint_ijk = __daxGetPointIJK(pid, dims);
  int4 point_ijk = (int4)(upoint_ijk.x, upoint_ijk.y, upoint_ijk.z, 0);
  daxUInt3 cell_dims;
  cell_dims.xyz = dims.xyz - as_daxUInt3(1, 1, 1).xyz;

  int4 offsets[8];
  offsets[0] = (int4)(-1,0,0, 0);
  offsets[1] = (int4)(-1,-1,0, 0);
  offsets[2] = (int4)(-1,-1,-1, 0);
  offsets[3] = (int4)(-1,0,-1, 0);
  offsets[4] = (int4)(0 , 0, 0, 0);
  offsets[5] = (int4)(0,-1,0,0);
  offsets[6] = (int4)(0,-1,-1, 0);
  offsets[7] = (int4)(0,0,-1, 0);

  int4 cell_ijk;
  int count = 0;
  for (uint j=0; j < 8; j++)
    {
    cell_ijk.xyz = point_ijk.xyz + offsets[j].xyz;
    if (cell_ijk.x < 0 || cell_ijk.x >= cell_dims.x ||
      cell_ijk.y < 0 || cell_ijk.y >= cell_dims.y ||
      cell_ijk.z < 0 || cell_ijk.z >= cell_dims.z)
      {
      continue;
      }
    cell_ids[count] = cell_ijk.x +
      cell_ijk.y * cell_dims.x + cell_ijk.z * cell_dims.x * cell_dims.y;
    count++;
    }
  return count;
}


daxFloat3 __daxGetArrayValue3(const daxWork* work, const daxArray* array)
{
  daxFloat3 retval = as_daxFloat3(0, 0, 0);
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
      retval.xyz = array->TempResultF.xyz;
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
    daxUInt3 dims = __daxGetDims(&imageData);
    // assume non-zero dims for now.
    daxUInt3 xyz; 
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

daxFloat __daxGetArrayValue(const daxWork* work, const daxArray* array)
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

void daxSetArrayValue(const daxWork* work, daxArray* output, const daxFloat scalar)
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

void daxSetArrayValue3(const daxWork* work, daxArray* output, const daxFloat3 scalar)
{
  if (output->Core->Type == ARRAY_TYPE_IRREGULAR)
    {
    if (output->OutputDataF != 0)
      {
      // indicates we are writing to a global out-array.
      //printf("%d == %f, \t",work->ElementID, scalar);
      output->OutputDataF[3*work->ElementID] = scalar.x;
      output->OutputDataF[3*work->ElementID+1] = scalar.y;
      output->OutputDataF[3*work->ElementID+2] = scalar.z;
      }
    else if (output->OutputDataF == 0)
      {
      // we are setting value for an intermediate array.
      // simply save the value in local memory.
      output->TempResultF.xyz = scalar.xyz;
      }
    }
}


// API for access topology.
struct __daxConnectedComponent
{
  daxIdType Id;
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

daxIdType daxGetNumberOfElements(const daxConnectedComponent* element)
{
  // For now, we assume that connections array is never generated, but a global
  // input array. Connections array will start being generated once we start
  // dealing with topology changing algorithms.
  daxIdType val = 3;
  switch (element->ConnectionsArray->Core->Type)
    {
  case ARRAY_TYPE_IMAGE_CELL:
    val = 8; // assume 3D cells for now.
    break;

  case ARRAY_TYPE_IMAGE_LINK:
    // this is non-constant even for voxel.
    // need to compute the number of cells the point belongs to.
      {
      daxImageDataData imageData =
        *((daxImageDataData*)(element->ConnectionsArray->InputDataF));
      daxUInt3 dims = __daxGetDims(&imageData);
      daxIdType pid = element->Id;
      daxUInt temp[8];
      val = __daxGetStructuredPointCells(pid, dims, temp);
      }
    break;
    }
  //printf("daxGetNumberOfElements case not handled.");
  return val;
}

void daxGetWorkForElement(const daxConnectedComponent* element,
  daxIdType index, daxWork* element_work)
{
  printf ("daxGetWorkForElement %d\n", index);
  switch (element->ConnectionsArray->Core->Type)
    {
  case ARRAY_TYPE_IMAGE_LINK:
    // given the point, return the cell-id containing that point.
      {
      daxImageDataData imageData =
        *((daxImageDataData*)(element->ConnectionsArray->InputDataF));
      daxUInt3 dims = __daxGetDims(&imageData);
      daxIdType pid = element->Id;
      daxUInt cellids[8];
      daxUInt num_cells = __daxGetStructuredPointCells(pid, dims, cellids);
      element_work->ElementID = cellids[index];
      }
    break;

  case ARRAY_TYPE_IMAGE_CELL:
      {
      daxImageDataData imageData =
        *((daxImageDataData*)(element->ConnectionsArray->InputDataF));
      daxUInt3 dims = __daxGetDims(&imageData);
      // assume non-zero dims for now.
      daxUInt3 ijk = __daxGetCellIJK(element->Id, dims);
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

daxFloat3 daxGetCellDerivative(const daxConnectedComponent* element,
  daxIdType subid, const daxFloat3 pcords, const daxFloat* scalars)
{
  uchar type = element->ConnectionsArray->Core->Type;
  switch (type)
    {
  case ARRAY_TYPE_IMAGE_CELL:
      {
      daxImageDataData imageData =
        *((daxImageDataData*)(element->ConnectionsArray->InputDataF));
      daxFloat3 spacing = as_daxFloat3(
        imageData.Spacing[0], imageData.Spacing[1], imageData.Spacing[2]);

      daxFloat3 rm_sm_tm;
      rm_sm_tm = as_daxFloat3(1, 1, 1) - pcords;

      float8 derivs[3];
      derivs[0].s0 = -rm_sm_tm.y*rm_sm_tm.z;
      derivs[0].s1 = rm_sm_tm.y*rm_sm_tm.z;
      derivs[0].s2 = -pcords.y*rm_sm_tm.z;
      derivs[0].s3 = pcords.y*rm_sm_tm.z;

      derivs[0].s4 = -rm_sm_tm.y*pcords.z;
      derivs[0].s5 = rm_sm_tm.y*pcords.z;

      derivs[0].s6 = -pcords.y*pcords.z;
      derivs[0].s7 = pcords.y*pcords.z;

      // s derivatives
      derivs[1].s0 = -rm_sm_tm.x*rm_sm_tm.z;
      derivs[1].s1 = -pcords.x*rm_sm_tm.z;
      derivs[1].s2 = rm_sm_tm.x*rm_sm_tm.z;
      derivs[1].s3 = pcords.x*rm_sm_tm.z;
      derivs[1].s4 = -rm_sm_tm.x*pcords.z;
      derivs[1].s5 = -pcords.x*pcords.z;
      derivs[1].s6 = rm_sm_tm.x*pcords.z;
      derivs[1].s7 = pcords.x*pcords.z;

      // t derivatives
      derivs[2].s0 = -rm_sm_tm.x*rm_sm_tm.y;
      derivs[2].s1 = -pcords.x*rm_sm_tm.y;
      derivs[2].s2 = -rm_sm_tm.x*pcords.y;
      derivs[2].s3 = -pcords.x*pcords.y;
      derivs[2].s4 = rm_sm_tm.x*rm_sm_tm.y;
      derivs[2].s5 = pcords.x*rm_sm_tm.y;
      derivs[2].s6 = rm_sm_tm.x*pcords.y;
      derivs[2].s7 = pcords.x*pcords.y;

      float8 scalars8 = vload8(0, scalars);
      float8 sum = derivs[0] * scalars8;

      daxFloat3 all_sum;
      all_sum.x = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 +
        sum.s6 + sum.s7;
      sum = derivs[1] * scalars8;
      all_sum.y = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 +
        sum.s6 + sum.s7;
      sum = derivs[2] * scalars8;
      all_sum.z = sum.s0 + sum.s1 + sum.s2 + sum.s3 + sum.s4 + sum.s5 +
        sum.s6 + sum.s7;
      return all_sum / spacing;
      }

    break;

  default:
    printf("daxGetWorkForElement case not handled.");
    }
  return as_daxFloat3(type, 0, 0);
}
