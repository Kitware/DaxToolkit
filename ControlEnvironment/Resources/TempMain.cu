
#include "DaxDataModel.cu"

__global__ void Execute(_DaxArray* arrays, int num_arrays, int *output)
{
  int i = threadIdx.x; 
  output[i] = arrays[i].GetNumberOfTuples();
}

#include <iostream>
using namespace std;

int main()
{
  _DaxArray h_arrays[10], *d_arrays;
  int *d_output, h_output[10];
  for (int cc=0; cc < 10; cc++)
    {
    h_arrays[cc].SetNumberOfTuples(cc*2);
    }
  cudaMalloc(&d_output, sizeof(int)*10);
  cudaMalloc(&d_arrays, sizeof(_DaxArray)*10);
  cudaMemcpy(d_arrays, h_arrays, sizeof(_DaxArray)*10, cudaMemcpyHostToDevice);
  Execute<<<1, 10>>>(d_arrays, 10, d_output);
  cudaMemcpy(h_output, d_output, sizeof(int)*10, cudaMemcpyDeviceToHost);
  cudaFree(d_output);
  cudaFree(d_arrays);
  for (int cc=0; cc < 10; cc++)
    {
    cout << h_output[cc] << endl;
    }
}
