/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxAttributeKeyBase_h
#define __daxAttributeKeyBase_h

#include "daxObject.h"

class daxAttributeKeyBase
{
public:
  daxAttributeKeyBase(const char* location, const char* key);
  virtual ~daxAttributeKeyBase();

  /// Returns the location for the key.
  const std::string& GetLocation() const
    { return this->Location; }

  /// Returns the name for the key.
  const std::string& GetName() const
    { return this->Name; }

private:
  std::string Location;
  std::string Name;
};

#define daxDefineKey(classname, keyname, type, container)\
  daxAttributeKey<type, container>* classname::keyname() { \
    static daxAttributeKey<type, container> __key(#classname, #keyname);\
    return &__key; }

#endif
