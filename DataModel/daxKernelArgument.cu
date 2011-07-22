/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxKernelArgument.h"

//-----------------------------------------------------------------------------
daxKernelArgument::daxKernelArgument()
{
}

//-----------------------------------------------------------------------------
daxKernelArgument::~daxKernelArgument()
{
  // release all cuda-memories allocated for the arrays.
  thrust::host_vector<DaxDataArray>::iterator iter;
  thrust::host_vector<DaxDataArray> host_arrays = this->Arrays;
  for (iter = host_arrays.begin(); iter != host_arrays.end(); ++iter)
    {
    cudaFree((*iter).RawData);
    }
  this->Arrays.clear();
}

//-----------------------------------------------------------------------------
const DaxKernelArgument& daxKernelArgument::Get()
{
  this->Argument.NumberOfArrays = this->Arrays.size();
  if (this->Arrays.size() > 0)
    {
    this->Argument.Arrays = thrust::raw_pointer_cast(&this->Arrays[0]);
    }
  else
    {
    this->Argument.Arrays = NULL;
    }
  return this->Argument;
}
