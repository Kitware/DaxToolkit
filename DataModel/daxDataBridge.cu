/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataBridge.h"

#include "daxDataSet.h"
#include "DaxDataArray.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <map>
class daxDataBridge::daxInternals
{
  std::vector<daxDataSetPtr> Inputs;
  std::vector<daxDataSetPtr> Intermediates;
  std::vector<daxDataSetPtr> Outputs;
  std::map<daxDataArray*, int> Arrays;
};

//-----------------------------------------------------------------------------
daxDataBridge::daxDataBridge()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxDataBridge::~daxDataBridge()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddInputData(daxDataSetPtr dataset)
{
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddIntermediateData(daxDataSetPtr dataset)
{
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddOutputData(daxDataSetPtr dataset)
{
}

//-----------------------------------------------------------------------------
//void daxDataBridge::Upload()
//{
//  // * Upload all daxDataArray's.
//  std::vector<daxDataSetPtr>::iterator iter;
//  for (iter = this->Internals->Inputs.begin();
//    iter != this->Internals->Inputs.end(); ++iter)
//    {
//    
//    }
//
//}

