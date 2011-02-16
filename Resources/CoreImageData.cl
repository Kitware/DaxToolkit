// OpenCL Code.
#define __GLOBAL_DATA_ITERATOR 0

// opaque_data_type is a data-structure that contains enough information about
// the mesh to enable location/iterating over cells/points as applicable.
typedef struct __attribute__( (packed endian(host) ) )
{
  // xmin, xmax, ymin, ymax etc.
  int Extents[6];

  // x,y,z.
  float Spacing[3];
} opaque_data_type;

typedef struct
{
  __global float * global_data_pointer;
  __local float * local_data_pointer;
} opaque_data_handle;

typedef struct
{
  const opaque_data_type* data_ptr;
  uint start_index[3];
  uint current_offset[3];
  uint end_offset[3];
  int type;
} PointIterator;

// Initializes "iterator" argument as a point iterator to iterate over global memory.
void point_iterator(PointIterator* iterator, const opaque_data_type data_handle)
{
  iterator->data_ptr = &data_handle;
  iterator->type = __GLOBAL_DATA_ITERATOR;
  iterator->start_index[0] = get_global_id(0);
  iterator->start_index[1] = get_global_id(1);
  iterator->start_index[2] = get_global_id(2);
  iterator->end_offset[0] = 1;
  iterator->end_offset[1] = 1;
  iterator->end_offset[2] = 1;
}

void begin_point(PointIterator *iter)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    iter->current_offset[0] = 0;
    iter->current_offset[1] = 0;
    iter->current_offset[2] = 0;
    }
}

bool is_done_point(PointIterator* iter)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    return (
      (iter->current_offset[0] >= iter->end_offset[0]) ||
      (iter->current_offset[1] >= iter->end_offset[1]) ||
      (iter->current_offset[2] >= iter->end_offset[2]));
    }

  return true;
}

void next_point(PointIterator* iter)
{
  int4 offsets = (int4)(
    iter->current_offset[0],
    iter->current_offset[1],
    iter->current_offset[2], 0);
  int4 endoffsets = (int4)(
    iter->end_offset[0],
    iter->end_offset[1],
    iter->end_offset[2], 0);

  if (offsets[0] < endoffsets[0])
    {
    offsets[0] += 1;
    if (offsets[0] < endoffsets[0])
      {
      return;
      }
    offsets[0] = 0;
    }
  if (offsets[1] < endoffsets[1])
    {
    offsets[1] += 1;
    if (offsets[1] < endoffsets[1])
      {
      return;
      }
    offsets[1] = 0;
    }
  offsets[2] += 1;
}

uint __flat_index(uint4 dims, uint4 ijk)
{
  return (ijk.z*dims.y*dims.x + ijk.y * dims.x + ijk.x);
}

__global float * get_data_reference_point(PointIterator* iter,
  opaque_data_handle* data_handle)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    uint4 dims;
    dims.x = iter->data_ptr->Extents[1] - iter->data_ptr->Extents[0] + 1;
    dims.y = iter->data_ptr->Extents[3] - iter->data_ptr->Extents[2] + 1;
    dims.z = iter->data_ptr->Extents[5] - iter->data_ptr->Extents[4] + 1;
    uint4 ijk;
    ijk.x = iter->start_index[0] + iter->current_offset[0];
    ijk.y = iter->start_index[1] + iter->current_offset[1];
    ijk.z = iter->start_index[2] + iter->current_offset[2];
    uint flat_offset = __flat_index(dims, ijk);
    return data_handle->global_data_pointer + flat_offset;
    }

  // something to the following effect.
 //return &data_handle->global_data_pointer[iter->current_offset];
 return &data_handle->global_data_pointer[0];
}

void global_data_handle(opaque_data_handle* handle , __global float * buffer)
{
  handle->global_data_pointer = buffer;
}
