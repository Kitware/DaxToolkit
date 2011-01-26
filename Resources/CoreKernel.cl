/// Body for the entry point for the openCL code.
/// Keywords:
/// * body - the actual kernel code.
__kernel void main(
  __global const opaque_iterator_type * opaque_data_ptr,
  __global const float * input_array,
  __global float * output_array)
{
$body$
}
