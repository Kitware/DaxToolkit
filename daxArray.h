/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxArray_h
#define __daxArray_h

#include "daxObject.h"
#include "daxAttributeKey.h"

/// daxArray holds the actual data, positions, connections etc. that represent
/// the data being analyzed. daxArray supports named attributes.

class daxArray;
daxDefinePtrMacro(daxArray);

#define daxArrayDefineSetGets(x)\
  void Set(const daxAttributeKey<x, daxArray>* key, x value)\
    { key->Set(this, value); }\
  x Get(const daxAttributeKey<x, daxArray>* key) const \
    { return key->Get(this); }

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
  void SetShape(const int*);

  /// Get the shape.
  const int* GetShape() const
    { return this->Shape; }

  /// Set a named attribute.
  bool HasAttribute(const daxAttributeKeyBase* key);
  void SetAttribute(const daxAttributeKeyBase* key, daxObjectPtr value);
  daxObjectPtr GetAttribute(const daxAttributeKeyBase* key) const;

  bool Has(const daxAttributeKeyBase* key)
    { return this->HasAttribute(key); }
  /// convenience methods to set attributes with right type.
  daxArrayDefineSetGets(int);
  daxArrayDefineSetGets(daxArrayWeakPtr);

  /// Attribute used for ELEMENT_TYPE.
  static daxAttributeKey<int, daxArray>* ELEMENT_TYPE();

  /// Attribute used for indicating the referred array.
  static daxAttributeKey<daxArrayWeakPtr, daxArray>* REF();

  /// Attribute used for indicating the array being depended upon.
  static daxAttributeKey<daxArrayWeakPtr, daxArray>* DEP();

protected:
  int Rank;
  int* Shape;

private:
  daxDisableCopyMacro(daxArray);
  class daxInternals;
  daxInternals* Internals;
};

#endif
