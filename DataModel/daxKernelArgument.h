/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxKernelArgument_h
#define __daxKernelArgument_h

#include "daxObject.h"
#include "DaxKernelArgument.h"
#include "DaxDataArray.h"
#include "DaxDataSet.h"
#include <thrust/device_vector.h>
#include <map>

daxDeclareClass(daxKernelArgument);
daxDeclareClass(daxDataArray);

class daxKernelArgument : public daxObject
{
public:
  daxKernelArgument();
  virtual ~daxKernelArgument();
  daxTypeMacro(daxKernelArgument, daxObject);

  /// return the object that can be passed to the kernel.
  const DaxKernelArgument& Get();

private:
  friend class daxDataBridge;
  DaxKernelArgument Argument;
  thrust::device_vector<DaxDataArray> Arrays;
  thrust::device_vector<DaxDataSet> Datasets;
  std::map<daxDataArrayPtr, int> ArrayMap;
};

#endif
