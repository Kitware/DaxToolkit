/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataBridge.h"

#include "daxDataSet.h"

#include <thrust/host_vector.h>

class daxDataBridge::daxInternals
{
  std::vector<daxDataSetPtr> Inputs;
  std::vector<daxDataSetPtr> Intermediates;
  std::vector<daxDataSetPtr> Outputs;
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
