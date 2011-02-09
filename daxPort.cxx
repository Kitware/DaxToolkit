/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxPort.h"

//-----------------------------------------------------------------------------
daxPort::daxPort()
{
  this->Type = invalid;
  this->NumberOfComponents = 0;
}

//-----------------------------------------------------------------------------
daxPort::~daxPort()
{
}

//-----------------------------------------------------------------------------
std::string daxPort::GetName() const
{
  return this->Name;
}

//-----------------------------------------------------------------------------
void daxPort::SetName(const std::string &name)
{
  this->Name = name;
}

//-----------------------------------------------------------------------------
int daxPort::GetType() const
{
  return this->Type;
}

//-----------------------------------------------------------------------------
void daxPort::SetType(int type)
{
  this->Type = type;
}

//-----------------------------------------------------------------------------
int daxPort::GetNumberOfComponents() const
{
  return this->NumberOfComponents;
}

//-----------------------------------------------------------------------------
void daxPort::SetNumberOfComponents(int num)
{
  this->NumberOfComponents = num;
}

//-----------------------------------------------------------------------------
bool daxPort::CanSourceFrom(const daxPort* sourcePort) const
{
  if (sourcePort->GetNumberOfComponents() != this->GetNumberOfComponents())
    {
    return false;
    }
  if (sourcePort->GetType() == this->GetType())
    {
    return true;
    }
  if (sourcePort->GetType() == any_array)
    {
    if (this->GetType() == float_)
      {
      return false;
      }
    return true;
    }
  if (this->GetType() == any_array)
    {
    if (sourcePort->GetType() == float_)
      {
      return false;
      }
    return true;
    }
  return false;
}
