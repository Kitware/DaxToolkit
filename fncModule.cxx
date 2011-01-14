/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncModule.h"

#include "fncPort.h"

#include <vector>

class fncModule::fncInternals
{
public:
  typedef std::vector<fncPortPtr> PortsCollectionType;
  PortsCollectionType InputPorts;
  PortsCollectionType OutputPorts;
};

//-----------------------------------------------------------------------------
fncModule::fncModule()
{
  this->Internals = new fncInternals();
}

//-----------------------------------------------------------------------------
fncModule::~fncModule()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void fncModule::SetNumberOfInputs(size_t count)
{
  this->Internals->InputPorts.resize(count);
}

//-----------------------------------------------------------------------------
void fncModule::SetInputPort(size_t index, fncPortPtr port)
{
  if (this->Internals->InputPorts.size() <= index)
    {
    this->Internals->InputPorts.resize(index + 1);
    }

  this->Internals->InputPorts[index] = port;
}

//-----------------------------------------------------------------------------
void fncModule::SetNumberOfOutputs(size_t count)
{
  this->Internals->OutputPorts.resize(count);
}

//-----------------------------------------------------------------------------
void fncModule::SetOutputPort(size_t index, fncPortPtr port)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    this->Internals->OutputPorts.resize(index + 1);
    }

  this->Internals->OutputPorts[index] = port;
}

//-----------------------------------------------------------------------------
size_t fncModule::GetNumberOfInputs()
{
  return this->Internals->InputPorts.size();
}

//-----------------------------------------------------------------------------
std::string fncModule::GetInputPortName(size_t index)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return std::string();
    }

  return this->Internals->InputPorts[index]->GetName();
}

//-----------------------------------------------------------------------------
fncPortPtr fncModule::GetInputPort(size_t index)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return fncPortPtr();
    }

  return this->Internals->InputPorts[index];
}

//-----------------------------------------------------------------------------
fncPortPtr fncModule::GetInputPort(const std::string& portname)
{
  fncInternals::PortsCollectionType::iterator iter;
  for (iter = this->Internals->InputPorts.begin();
    iter != this->Internals->InputPorts.end();
    iter++)
    {
    if (*iter && (*iter)->GetName() == portname)
      {
      return *iter;
      }
    }
  return fncPortPtr();
}

//-----------------------------------------------------------------------------
size_t fncModule::GetNumberOfOutputs()
{
  return this->Internals->OutputPorts.size();
}

//-----------------------------------------------------------------------------
std::string fncModule::GetOutputPortName(size_t index)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return std::string();
    }

  return this->Internals->OutputPorts[index]->GetName();
}

//-----------------------------------------------------------------------------
fncPortPtr fncModule::GetOutputPort(size_t index)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return fncPortPtr();
    }

  return this->Internals->OutputPorts[index];
}

//-----------------------------------------------------------------------------
fncPortPtr fncModule::GetOutputPort(const std::string& portname)
{
  fncInternals::PortsCollectionType::iterator iter;
  for (iter = this->Internals->OutputPorts.begin();
    iter != this->Internals->OutputPorts.end();
    iter++)
    {
    if (*iter && (*iter)->GetName() == portname)
      {
      return *iter;
      }
    }
  return fncPortPtr();
}
