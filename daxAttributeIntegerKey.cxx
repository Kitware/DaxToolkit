/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxAttributeIntegerKey.h"

#include "daxArray.h"

class daxAttributeIntegerKey::daxKeyValue : public daxObject
{
public:
  daxKeyValue(int value) { this->Value = value; }
  int Value;
};


//-----------------------------------------------------------------------------
daxAttributeIntegerKey::daxAttributeIntegerKey(
  const char* l, const char* n) : Superclass(l, n)
{
}

//-----------------------------------------------------------------------------
daxAttributeIntegerKey::~daxAttributeIntegerKey()
{
}

//-----------------------------------------------------------------------------
void daxAttributeIntegerKey::Set(daxArray* container, int value) const
{
  daxObjectPtr _value(new daxKeyValue(value));
  container->SetAttribute(this, _value);
}

//-----------------------------------------------------------------------------
int daxAttributeIntegerKey::Get(const daxArray* container) const
{
  daxObjectPtr _value = container->GetAttribute(this);
  if (_value.get())
    {
    daxKeyValue* kv = dynamic_cast<daxKeyValue*>(_value.get());
    return kv? kv->Value : 0;
    }
  return 0;
}
