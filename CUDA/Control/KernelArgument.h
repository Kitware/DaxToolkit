/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_KernelArgument_h
#define __dax_cuda_cont_KernelArgument_h

#include "Core/Control/Object.h"

#include "CUDA/Common/KernelArgument.h"
#include "Core/Common/DataArray.h"
#include "Core/Common/DataSet.h"

#include <map>
#include <vector>

namespace dax { namespace cont {
  daxDeclareClass(DataArray);
}}

namespace dax { namespace cuda { namespace cont {

daxDeclareClass(DataBridge);
daxDeclareClass(KernelArgument);

class KernelArgument : public dax::core::cont::Object
{
public:
  KernelArgument();
  virtual ~KernelArgument();
  daxTypeMacro(KernelArgument, dax::core::cont::Object);

  /// return the object that can be passed to the kernel.
  const dax::cuda::KernelArgument& Get();

  void SetDataSets(const std::vector<dax::core::DataSet>& datasets);
  void SetArrays(const std::vector<dax::core::DataArray>& arrays);
  void SetArrayMap(const std::map<dax::cont::DataArrayPtr, int> array_map);

protected:
  friend class dax::cuda::cont::DataBridge;
  std::vector<dax::core::DataArray> HostArrays;
  std::vector<dax::core::DataSet> HostDatasets;
  std::map<dax::cont::DataArrayPtr, int> ArrayMap;


private:
  dax::cuda::KernelArgument Argument;
  dax::core::DataArray* DeviceArrays;
  dax::core::DataSet* DeviceDatasets;
};
}}}

#endif
