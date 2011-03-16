/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxArray.h"

#include <map>
class daxArray::daxInternals
{
public:
  typedef std::map<const daxAttributeKeyBase*, daxObjectPtr> AttributeMapType;
  AttributeMapType AttributeMap;
};

daxDefineKey(daxArray, ELEMENT_TYPE, int, daxArray);
daxDefineKey(daxArray, REF, daxArrayWeakPtr, daxArray);
daxDefineKey(daxArray, DEP, daxArrayWeakPtr, daxArray);
//-----------------------------------------------------------------------------
daxArray::daxArray()
{
  this->Rank = 0;
  this->Shape = NULL;
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxArray::~daxArray()
{
  delete [] this->Shape;
  this->Shape = NULL;
  delete this->Internals;
}

//-----------------------------------------------------------------------------
void daxArray::SetShape(const int* shape)
{
  delete [] this->Shape;
  this->Shape = NULL;
  if (this->Rank > 0)
    {
    this->Shape = new int[this->Rank];
    for (int cc=0; cc < this->Rank; cc++)
      {
      this->Shape[cc] = shape[cc];
      }
    }
}

//-----------------------------------------------------------------------------
void daxArray::SetAttribute(const daxAttributeKeyBase* key, daxObjectPtr value)
{
  this->Internals->AttributeMap[key] = value;
}

//-----------------------------------------------------------------------------
bool daxArray::HasAttribute(const daxAttributeKeyBase* key)
{
  return (this->Internals->AttributeMap.find(key) !=
    this->Internals->AttributeMap.end());
}

//-----------------------------------------------------------------------------
daxObjectPtr daxArray::GetAttribute(const daxAttributeKeyBase* key) const
{
  daxInternals::AttributeMapType::iterator iter =
    this->Internals->AttributeMap.find(key);
  if (iter != this->Internals->AttributeMap.end())
    {
    return iter->second;
    }

  return daxObjectPtr();
}
