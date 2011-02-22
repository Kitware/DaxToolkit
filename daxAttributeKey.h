/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxAttributeKey_h
#define __daxAttributeKey_h

#include "daxAttributeKeyBase.h"
template <class T, class C>
class daxAttributeKey : public daxAttributeKeyBase
{
  class ValueClass;
public:
  daxAttributeKey(const char* l, const char* n): Superclass(l, n) {};
  daxTypeMacro(daxAttributeKey, daxAttributeKeyBase);

  void Set(C* container, T value) const
    {
    daxObjectPtr _value(new ValueClass(value));
    container->SetAttribute(this, _value);
    }

  T Get(const C* container) const
    {
    daxObjectPtr _value = container->GetAttribute(this);
    if (_value.get())
      {
      ValueClass* kv = dynamic_cast<ValueClass*>(_value.get());
      return kv? kv->Value : T();
      }
    return T();
    }
private:
  class ValueClass : public daxObject
    {
  public:
    ValueClass(T value) { this->Value = value; }
    T Value;
    };
};

#endif
