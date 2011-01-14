/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncExecutive.h"

#include "fncPort.h"
#include "fncModule.h"

//-----------------------------------------------------------------------------
fncExecutive::fncExecutive()
{
}

//-----------------------------------------------------------------------------
fncExecutive::~fncExecutive()
{
}

//-----------------------------------------------------------------------------
void fncExecutive::Connect(
  fncModulePtr sourceModule, const std::string& sourcename,
  fncModulePtr sinkModule, const std::string& sinkname)
{
  this->Connect(sourceModule, sourceModule->GetOutputPort(sourcename),
    sinkModule, sinkModule->GetInputPort(sinkname));
}

//-----------------------------------------------------------------------------
void fncExecutive::Connect(
  fncModulePtr sourceModule, fncPortPtr sourcePort,
  fncModulePtr sinkModule, fncPortPtr sinkPort)
{
}

//-----------------------------------------------------------------------------
bool fncExecutive::Execute()
{
  return false;
}
