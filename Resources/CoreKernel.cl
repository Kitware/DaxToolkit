// OpenCL Code.

/// Body for the entry point for the openCL code.
/// Keywords:
/// * body - the actual kernel code.
__kernel void main(
  __global const opaque_data_type opaque_data_pointer,
  __global float * input_array,
  __global float * output_array)
{
  opaque_data_handle inputHandle = global_data_handle(input_array);
  opaque_data_handle outputHandle = global_data_handle(output_array);


$body$
}
