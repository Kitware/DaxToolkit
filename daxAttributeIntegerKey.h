/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxAttributeIntegerKey_h
#define __daxAttributeIntegerKey_h

#include "daxAttributeKey.h"

class daxArray;

class daxAttributeIntegerKey : public daxAttributeKey
{
public:
  daxAttributeIntegerKey(const char* l, const char* n);
  virtual ~daxAttributeIntegerKey();
  daxTypeMacro(daxAttributeIntegerKey, daxAttributeKey);

  void Set(daxArray* container, int value) const;
  int Get(const daxArray* container) const;

private:
  class daxKeyValue;
};

daxDefinePtrMacro(daxAttributeIntegerKey);

#endif
