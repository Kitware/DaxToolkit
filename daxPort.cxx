/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxPort.h"

//-----------------------------------------------------------------------------
daxPort::daxPort()
{
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
bool daxPort::CanSourceFrom(const daxPort* sourcePort) const
{
  if (sourcePort == this)
    {
    return false;
    }
  return true;
}
