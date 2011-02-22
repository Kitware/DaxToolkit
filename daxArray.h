/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxArray_h
#define __daxArray_h

#include "daxObject.h"

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

protected:
  int Rank;
  int* Shape;

private:
  daxDisableCopyMacro(daxArray);
};

daxDefinePtrMacro(daxArray);
#endif
