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
  uint start_index;
  uint current_offset;
  uint end_offset;
  int type;
} PointIterator;

// Initializes "iterator" argument as a point iterator to iterate over global memory.
void point_iterator(PointIterator* iterator, const opaque_data_type data_handle)
{
  iterator->data_ptr = &data_handle;
  iterator->type = __GLOBAL_DATA_ITERATOR;
  iterator->start_index = get_global_id(0);
  iterator->end_offset = 1;
  iterator->current_offset = 0;
}

void begin_point(PointIterator *iter)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    iter->current_offset = 0;
    }
}

bool is_done_point(PointIterator* iter)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    return (iter->current_offset >= iter->end_offset);
    }

  return true;
}

void next_point(PointIterator* iter)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    iter->current_offset++;
    }
}

float get_value_point(PointIterator* iter, opaque_data_handle* data_handle)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    uint flat_offset = iter->start_index + iter->current_offset;
    return data_handle->global_data_pointer[flat_offset];
    }
}

void set_value_point(
  PointIterator* iter, opaque_data_handle* data_handle, float value)
{
  if (iter->type == __GLOBAL_DATA_ITERATOR)
    {
    uint flat_offset = iter->start_index + iter->current_offset;
    data_handle->global_data_pointer[flat_offset] = value;
    }
}

void global_data_handle(opaque_data_handle* handle , __global float * buffer)
{
  handle->global_data_pointer = buffer;
}
