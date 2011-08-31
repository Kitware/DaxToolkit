/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cuda_cont_internal_DataBridge_h
#define __dax_cuda_cont_internal_DataBridge_h

#include <dax/internal/DataArray.h>
#include <dax/cont/internal/Object.h>

namespace dax { namespace cont {
  daxDeclareClass(DataSet);
  daxDeclareClass(DataArray);
}}

namespace dax { namespace cuda { namespace cont { namespace internal {

daxDeclareClass(DataBridge);
daxDeclareClass(KernelArgument);

/// DataBridge is used to transfer host-memory data objects to the device.
/// This class is not meant to be used by users of the ControlEnvironment. It
/// will be used internally to invoke the kernel once the ControlEnvironment is
/// mature enough.
class DataBridge : public dax::cont::internal::Object
{
public:
  DataBridge();
  virtual ~DataBridge();
  daxTypeMacro(DataBridge, dax::cont::internal::Object);

  /// Add an input dataset.
  void AddInputData(dax::cont::DataSetPtr dataset);

  /// Add an intermediate dataset.
  void AddIntermediateData(dax::cont::DataSetPtr dataset);

  /// Add output dataset.
  void AddOutputData(dax::cont::DataSetPtr dataset);

  /// makes it possible to pass this class as an argument to a CUDA kernel.
  dax::cuda::cont::internal::KernelArgumentPtr Upload() const;

  /// downloads the results.
  bool Download(dax::cuda::cont::internal::KernelArgumentPtr argument) const;

protected:
  virtual dax::internal::DataArray UploadArray(
    dax::cont::DataArrayPtr host_array, bool copy_heavy_data) const;

  virtual bool DownloadArray(
    dax::cont::DataArrayPtr host_array,
    const dax::internal::DataArray& device_array) const;

private:
  daxDisableCopyMacro(DataBridge)

  class daxInternals;
  daxInternals* Internals;
};

}}}}

#endif
