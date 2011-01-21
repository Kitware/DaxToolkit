/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncPort.h"

//-----------------------------------------------------------------------------
fncPort::fncPort()
{
  this->Type = invalid;
  this->NumberOfComponents = 0;
}

//-----------------------------------------------------------------------------
fncPort::~fncPort()
{
}

//-----------------------------------------------------------------------------
std::string fncPort::GetName()
{
  return this->Name;
}

//-----------------------------------------------------------------------------
void fncPort::SetName(const std::string &name)
{
  this->Name = name;
}

//-----------------------------------------------------------------------------
int fncPort::GetType()
{
  return this->Type;
}

//-----------------------------------------------------------------------------
void fncPort::SetType(int type)
{
  this->Type = type;
}

//-----------------------------------------------------------------------------
int fncPort::GetNumberOfComponents()
{
  return this->NumberOfComponents;
}

//-----------------------------------------------------------------------------
void fncPort::SetNumberOfComponents(int num)
{
  this->NumberOfComponents = num;
}

//-----------------------------------------------------------------------------
bool fncPort::CanSourceFrom(fncPort* sourcePort)
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
