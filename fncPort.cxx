/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "fncPort.h"

//-----------------------------------------------------------------------------
fncPort::fncPort()
{
  this->Type = 0;
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
