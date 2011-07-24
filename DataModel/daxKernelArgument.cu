/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxKernelArgument.h"

#include <assert.h>

//-----------------------------------------------------------------------------
daxKernelArgument::daxKernelArgument()
{
  this->DeviceDatasets = NULL;
  this->DeviceArrays = NULL;
}

//-----------------------------------------------------------------------------
daxKernelArgument::~daxKernelArgument()
{
  // release all cuda-memories allocated for the arrays.
  //thrust::host_vector<DaxDataArray>::iterator iter;
  //thrust::host_vector<DaxDataArray> host_arrays = this->Arrays;
  //for (iter = host_arrays.begin(); iter != host_arrays.end(); ++iter)
  //  {
  //  cudaFree((*iter).RawData);
  //  }
  //this->Arrays.clear();
  cout << "Need to free cuda-memory" << endl;
}

//-----------------------------------------------------------------------------
void daxKernelArgument::SetDataSets(const std::vector<DaxDataSet>& datasets)
{
  this->HostDatasets = datasets;
}

//-----------------------------------------------------------------------------
void daxKernelArgument::SetArrays(const std::vector<DaxDataArray>& arrays)
{
  this->HostArrays = arrays;
}

//-----------------------------------------------------------------------------
void daxKernelArgument::SetArrayMap(const std::map<daxDataArrayPtr, int> array_map)
{
  this->ArrayMap = array_map;
}

//-----------------------------------------------------------------------------
const DaxKernelArgument& daxKernelArgument::Get()
{
  assert(this->HostArrays.size() > 0 && this->HostDatasets.size() > 0);

  cudaMalloc(&this->DeviceArrays,
    sizeof(DaxDataArray) * this->HostArrays.size());

  cudaMalloc(&this->DeviceDatasets,
    sizeof(DaxDataSet) * this->HostDatasets.size());

  cudaMemcpy(this->DeviceArrays, &this->HostArrays[0],
    sizeof(DaxDataArray) * this->HostArrays.size(),
    cudaMemcpyHostToDevice);

  cudaMemcpy(this->DeviceDatasets, &this->HostDatasets[0],
    sizeof(DaxDataSet) * this->HostDatasets.size(),
    cudaMemcpyHostToDevice);

  this->Argument.NumberOfDatasets = this->HostDatasets.size();
  this->Argument.NumberOfArrays = this->HostArrays.size();
  this->Argument.Arrays = this->DeviceArrays;
  this->Argument.Datasets = this->DeviceDatasets;
  return this->Argument;
}
