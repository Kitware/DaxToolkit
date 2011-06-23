/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxRegularArray_h
#define __daxRegularArray_h

/// daxRegularArray defines a regular array.

#include "daxArray.h"

template <class T>
class daxRegularArray : public daxArray
{
public:
  daxRegularArray();
  virtual ~daxRegularArray();
  daxTypeMacro(daxRegularArray, daxArray);

  /// Set the number of items.
  void SetNumberOfItems(size_t val)
    { this->NumberOfItems = val; }

  /// Get the number of items.
  size_t GetNumberOfItems() const
    { return this->NumberOfItems; }

  /// Set the origin.
  void SetOrigin(const T* origin);

  /// Get origin.
  const T* GetOrigin() const
    { return this->Origin; }

  /// Set the delta.
  void SetDelta(const T* delta);

  /// Get the delta.
  const T* GetDelta() const
    { return this->Delta; }

protected:
  T* Origin;
  T* Delta;
  size_t NumberOfItems;

private:
  daxDisableCopyMacro(daxRegularArray);
};

#include "daxRegularArray.txx"

#endif
