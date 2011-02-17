// OpenCL Code.

/// Body for the entry point for the openCL code.
/// Keywords:
/// * body - the actual kernel code.
__kernel void main(
  __global const opaque_data_type* opaque_data_pointer,
  __global const float * input_array,
  __global float * output_array)
{
  opaque_data_handle inputHandle;
  global_data_handle(&inputHandle, input_array);

  opaque_data_handle outputHandle;
  global_data_handle(&outputHandle, output_array);

  if (get_global_id(0) == 0)
    {
    // print some debugging information.
    printf("Extents: %d, %d, %d, %d, %d, %d \n",
      opaque_data_pointer->Extents[0],
      opaque_data_pointer->Extents[1],
      opaque_data_pointer->Extents[2],
      opaque_data_pointer->Extents[3],
      opaque_data_pointer->Extents[4],
      opaque_data_pointer->Extents[5]);
    printf("Spacing: %f, %f, %f\n",
      opaque_data_pointer->Spacing[0],
      opaque_data_pointer->Spacing[1],
      opaque_data_pointer->Spacing[2]);
    }
$body$
}
