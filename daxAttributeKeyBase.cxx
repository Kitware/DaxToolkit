/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxAttributeKeyBase.h"

//-----------------------------------------------------------------------------
daxAttributeKeyBase::daxAttributeKeyBase(const char* location, const char* name)
{
  assert(location && name);
  this->Location = location;
  this->Name = name;
}

//-----------------------------------------------------------------------------
daxAttributeKeyBase::~daxAttributeKeyBase()
{
}
