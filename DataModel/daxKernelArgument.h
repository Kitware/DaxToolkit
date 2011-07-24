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
#include <map>
#include <vector>

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

  void SetDataSets(const std::vector<DaxDataSet>& datasets);
  void SetArrays(const std::vector<DaxDataArray>& arrays);
  void SetArrayMap(const std::map<daxDataArrayPtr, int> array_map);

protected:
  friend class daxDataBridge;
  std::vector<DaxDataArray> HostArrays;
  std::vector<DaxDataSet> HostDatasets;
  std::map<daxDataArrayPtr, int> ArrayMap;


private:
  DaxKernelArgument Argument;
  DaxDataArray* DeviceArrays;
  DaxDataSet* DeviceDatasets;
};

#endif
