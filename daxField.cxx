/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxField.h"

#include "daxArray.h"
#include <map>
#include <assert.h>

class daxField::daxInternals
{
public:
  typedef std::map<std::string, daxArrayPtr> ComponentMapType;
  ComponentMapType ComponentMap;
};

//-----------------------------------------------------------------------------
daxField::daxField()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxField::~daxField()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void daxField::SetComponent(const char* name, daxArrayPtr array)
{
  assert(name && name[0] && array);
  this->Internals->ComponentMap[name] = array;
}

//-----------------------------------------------------------------------------
daxArrayPtr daxField::GetComponent(const char* name) const
{
  daxInternals::ComponentMapType::iterator iter =
    this->Internals->ComponentMap.find(name);
  if (iter != this->Internals->ComponentMap.end())
    {
    return iter->second;
    }
  return daxArrayPtr();
}

//-----------------------------------------------------------------------------
bool daxField::HasComponent(const char* name) const
{
  daxInternals::ComponentMapType::iterator iter =
    this->Internals->ComponentMap.find(name);
  return (iter != this->Internals->ComponentMap.end());
}
