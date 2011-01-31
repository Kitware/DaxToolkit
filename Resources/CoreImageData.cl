// OpenCL Code.
#define __DATA_ITERATOR 0

// opaque_data_type is a data-structure that contains enough information about
// the mesh to enable location/iterating over cells/points as applicable.
typedef struct __attribute__( (packed endian(host) ) )
{
  int Dimensions[3];
} opaque_data_type;

typedef struct
{
  __global float * global_data_pointer;
  __local float * local_data_pointer;
} opaque_data_handle;

typedef struct
{
  const opaque_data_type* data_ptr;
  uint current_offset;
  int type;
} PointIterator;

PointIterator point_iterator(const opaque_data_type data_handle)
{
  PointIterator iterator;
  iterator.data_ptr = &data_handle;
  iterator.current_offset = 0;
  iterator.type = __DATA_ITERATOR;
  return iterator;
}

void begin_point(PointIterator *iter)
{
  if (iter->type == __DATA_ITERATOR)
    {
    iter->current_offset= 0;
    }
}

bool is_done_point(PointIterator* iter)
{
  // check whether the current_offset is valid.
  return true;
}

void next_point(PointIterator* iter)
{
  iter->current_offset++;
}

__global float * get_data_reference_point(PointIterator* iter,
  opaque_data_handle* data_handle)
{
  // something to the following effect.
  return &data_handle->global_data_pointer[iter->current_offset];
}

opaque_data_handle global_data_handle(__global float * buffer)
{
  opaque_data_handle handle;
  handle.global_data_pointer = buffer;
  return handle;
}
