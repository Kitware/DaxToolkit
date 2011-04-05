/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxModule.h"

#include "daxPort.h"

#include <vector>

class daxModule::daxInternals
{
public:
  typedef std::vector<daxPortPtr> PortsCollectionType;
  PortsCollectionType InputPorts;
  PortsCollectionType OutputPorts;
};

//-----------------------------------------------------------------------------
daxModule::daxModule()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxModule::~daxModule()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void daxModule::SetNumberOfInputs(size_t count)
{
  this->Internals->InputPorts.resize(count);
}

//-----------------------------------------------------------------------------
void daxModule::SetInputPort(size_t index, daxPortPtr port)
{
  if (this->Internals->InputPorts.size() <= index)
    {
    this->Internals->InputPorts.resize(index + 1);
    }

  this->Internals->InputPorts[index] = port;
}

//-----------------------------------------------------------------------------
void daxModule::SetNumberOfOutputs(size_t count)
{
  this->Internals->OutputPorts.resize(count);
}

//-----------------------------------------------------------------------------
void daxModule::SetOutputPort(size_t index, daxPortPtr port)
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    this->Internals->OutputPorts.resize(index + 1);
    }

  this->Internals->OutputPorts[index] = port;
}

//-----------------------------------------------------------------------------
size_t daxModule::GetNumberOfInputs() const
{
  return this->Internals->InputPorts.size();
}

//-----------------------------------------------------------------------------
std::string daxModule::GetInputPortName(size_t index) const
{
  if (this->Internals->InputPorts.size() <= index)
    {
    return std::string();
    }

  return this->Internals->InputPorts[index]->GetName();
}

//-----------------------------------------------------------------------------
daxPortPtr daxModule::GetInputPort(size_t index) const
{
  if (this->Internals->InputPorts.size() <= index)
    {
    return daxPortPtr();
    }

  return this->Internals->InputPorts[index];
}

//-----------------------------------------------------------------------------
daxPortPtr daxModule::GetInputPort(const std::string& portname) const
{
  daxInternals::PortsCollectionType::iterator iter;
  for (iter = this->Internals->InputPorts.begin();
    iter != this->Internals->InputPorts.end();
    iter++)
    {
    if (*iter && (*iter)->GetName() == portname)
      {
      return *iter;
      }
    }
  return daxPortPtr();
}

//-----------------------------------------------------------------------------
size_t daxModule::GetNumberOfOutputs() const
{
  return this->Internals->OutputPorts.size();
}

//-----------------------------------------------------------------------------
std::string daxModule::GetOutputPortName(size_t index) const
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return std::string();
    }

  return this->Internals->OutputPorts[index]->GetName();
}

//-----------------------------------------------------------------------------
daxPortPtr daxModule::GetOutputPort(size_t index) const
{
  if (this->Internals->OutputPorts.size() <= index)
    {
    return daxPortPtr();
    }

  return this->Internals->OutputPorts[index];
}

//-----------------------------------------------------------------------------
daxPortPtr daxModule::GetOutputPort(const std::string& portname) const
{
  daxInternals::PortsCollectionType::iterator iter;
  for (iter = this->Internals->OutputPorts.begin();
    iter != this->Internals->OutputPorts.end();
    iter++)
    {
    if (*iter && (*iter)->GetName() == portname)
      {
      return *iter;
      }
    }
  return daxPortPtr();
}

#include <boost/algorithm/string/replace.hpp>
//-----------------------------------------------------------------------------
std::string daxModule::GetCleandupFunctorCode() const
{
  std::string code = this->GetFunctorCode();
  std::string format0 = 
    std::string("daxGetArrayValue3_")+this->GetModuleName();
  std::string format1= std::string("daxGetArrayValue_")+this->GetModuleName();

  boost::replace_all(code, "daxGetArrayValue3", "daxGetArray3Value");
  boost::replace_all(code, "daxGetArrayValue", format1);
  boost::replace_all(code, "daxGetArray3Value", format0);

  std::string header =
    std::string("float3 daxGetArrayValue3_")+this->GetModuleName() +
    "(const daxWork* work, const daxArray* array);\n" + 
    std::string("float daxGetArrayValue_")+this->GetModuleName() +
    "(const daxWork* work, const daxArray* array);\n";
  return (header + code);
}
