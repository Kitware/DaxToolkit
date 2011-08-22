/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_KernelArgument_h
#define __dax_cuda_KernelArgument_h

namespace dax { namespace core {
  class DataArray;
  class DataSet;
}}

namespace dax { namespace cuda {

/// KernelArgument is passed to the entry-point into the execution
/// environment. It encapsulates all information about the datasets used in the
/// pipeline.
class KernelArgument
{
public:
  dax::core::DataArray* Arrays;
  dax::core::DataSet* Datasets;
  int NumberOfArrays;
  int NumberOfDatasets;
};

}}

#endif
