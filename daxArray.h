/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxArray_h
#define __daxArray_h

#include "daxObject.h"
#include "daxAttributeKey.h"

class daxArray : public daxObject
{
public:
  daxArray();
  virtual ~daxArray();
  daxTypeMacro(daxArray, daxObject);

  /// Set the rank of the array.
  void SetRank(int rank)
    { this->Rank = rank; }

  /// Get the rank of the array.
  int GetRank() const
    { return this->Rank; }

  /// Set the shape.
  void SetShape(int*);

  /// Get the shape.
  const int* GetShape() const
    { return this->Shape; }

  /// Set a named attribute.
  bool HasAttribute(const daxAttributeKeyBase* key);
  void SetAttribute(const daxAttributeKeyBase* key, daxObjectPtr value);
  daxObjectPtr GetAttribute(const daxAttributeKeyBase* key) const;

  /// convenience methods to set attributes with right type.
  void Set(const daxAttributeKey<int, daxArray>* key, int value)
    { key->Set(this, value); }
  int Get(const daxAttributeKey<int, daxArray>* key) const
    { return key->Get(this); }

  /// Attribute used for ELEMENT_TYPE.
  static daxAttributeKey<int, daxArray>* ELEMENT_TYPE();

protected:
  int Rank;
  int* Shape;

private:
  daxDisableCopyMacro(daxArray);
  class daxInternals;
  daxInternals* Internals;
};

daxDefinePtrMacro(daxArray);
#endif
