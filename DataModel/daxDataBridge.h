/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataBridge_h
#define __daxDataBridge_h

#include "daxObject.h"

#ifndef SKIP_DOXYGEN
class daxDataSet;
daxDefinePtrMacro(daxDataSet);

class daxKernelArgument;
daxDefinePtrMacro(daxKernelArgument);

#endif

/// daxDataBridge is used to transfer host-memory data objects to the device.
/// This class is not meant to be used by users of the ControlEnvironment. It
/// will be used internally to invoke the kernel once the ControlEnvironment is
/// mature enough.
class daxDataBridge : public daxObject
{
public:
  daxDataBridge();
  virtual ~daxDataBridge();
  daxTypeMacro(daxDataBridge, daxObject);

  /// Add an input dataset.
  void AddInputData(daxDataSetPtr dataset);

  /// Add an intermediate dataset.
  void AddIntermediateData(daxDataSetPtr dataset);

  /// Add output dataset.
  void AddOutputData(daxDataSetPtr dataset);

  /// makes it possible to pass this class as an argument to a CUDA kernel.
  daxKernelArgumentPtr Upload() const;

  /// downloads the results.
  bool Download(daxKernelArgumentPtr argument) const;

private:
  daxDisableCopyMacro(daxDataBridge)

  class daxInternals;
  daxInternals* Internals;
};

/// declares daxDataBridgePtr
daxDefinePtrMacro(daxDataBridge)
#endif
