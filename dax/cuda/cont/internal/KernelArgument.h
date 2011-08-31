/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_KernelArgument_h
#define __dax_cuda_cont_internal_KernelArgument_h

#include <dax/cont/internal/Object.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/DataSet.h>

#include <dax/cuda/internal/KernelArgument.h>

#include <map>
#include <vector>

namespace dax { namespace cont {
  daxDeclareClass(DataArray);
}}

namespace dax { namespace cuda { namespace cont { namespace internal {

daxDeclareClass(DataBridge);
daxDeclareClass(KernelArgument);

class KernelArgument : public dax::cont::internal::Object
{
public:
  KernelArgument();
  virtual ~KernelArgument();
  daxTypeMacro(KernelArgument, dax::cont::internal::Object);

  /// return the object that can be passed to the kernel.
  const dax::cuda::internal::KernelArgument& Get();

  void SetDataSets(const std::vector<dax::internal::DataSet>& datasets);
  void SetArrays(const std::vector<dax::internal::DataArray>& arrays);
  void SetArrayMap(const std::map<dax::cont::DataArrayPtr, int> array_map);

protected:
  friend class dax::cuda::cont::internal::DataBridge;
  std::vector<dax::internal::DataArray> HostArrays;
  std::vector<dax::internal::DataSet> HostDatasets;
  std::map<dax::cont::DataArrayPtr, int> ArrayMap;


private:
  dax::cuda::internal::KernelArgument Argument;
  dax::internal::DataArray* DeviceArrays;
  dax::internal::DataSet* DeviceDatasets;
};
}}}}

#endif
