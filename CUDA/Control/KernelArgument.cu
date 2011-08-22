/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "CUDA/Control/KernelArgument.h"

#include <assert.h>

//-----------------------------------------------------------------------------
dax::cuda::cont::KernelArgument::KernelArgument()
{
  this->DeviceDatasets = NULL;
  this->DeviceArrays = NULL;
}

//-----------------------------------------------------------------------------
dax::cuda::cont::KernelArgument::~KernelArgument()
{
  // release all cuda-memories allocated for the arrays.
  //thrust::host_vector<dax::core::DataArray>::iterator iter;
  //thrust::host_vector<dax::core::DataArray> host_arrays = this->Arrays;
  //for (iter = host_arrays.begin(); iter != host_arrays.end(); ++iter)
  //  {
  //  cudaFree((*iter).RawData);
  //  }
  //this->Arrays.clear();
  using namespace std;
  cout << "Need to free cuda-memory" << endl;
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::KernelArgument::SetDataSets(const std::vector<dax::core::DataSet>& datasets)
{
  this->HostDatasets = datasets;
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::KernelArgument::SetArrays(const std::vector<dax::core::DataArray>& arrays)
{
  this->HostArrays = arrays;
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::KernelArgument::SetArrayMap(
  const std::map<dax::cont::DataArrayPtr, int> array_map)
{
  this->ArrayMap = array_map;
}

//-----------------------------------------------------------------------------
const dax::cuda::KernelArgument& dax::cuda::cont::KernelArgument::Get()
{
  assert(this->HostArrays.size() > 0 && this->HostDatasets.size() > 0);

  cudaMalloc(&this->DeviceArrays,
    sizeof(dax::core::DataArray) * this->HostArrays.size());

  cudaMalloc(&this->DeviceDatasets,
    sizeof(dax::core::DataSet) * this->HostDatasets.size());

  cudaMemcpy(this->DeviceArrays, &this->HostArrays[0],
    sizeof(dax::core::DataArray) * this->HostArrays.size(),
    cudaMemcpyHostToDevice);

  cudaMemcpy(this->DeviceDatasets, &this->HostDatasets[0],
    sizeof(dax::core::DataSet) * this->HostDatasets.size(),
    cudaMemcpyHostToDevice);

  this->Argument.NumberOfDatasets = this->HostDatasets.size();
  this->Argument.NumberOfArrays = this->HostArrays.size();
  this->Argument.Arrays = this->DeviceArrays;
  this->Argument.Datasets = this->DeviceDatasets;
  return this->Argument;
}
