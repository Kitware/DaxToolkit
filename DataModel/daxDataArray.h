/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataArray_h
#define __daxDataArray_h

#include "daxObject.h"

class DaxDataArray;

daxDeclareClass(daxDataArray);

/// daxDataArray is the abstract superclass for data array object containing
/// numeric data.
class daxDataArray : public daxObject
{
public:
  daxDataArray();
  virtual ~daxDataArray();
  daxTypeMacro(daxDataArray, daxObject);

  /// Get/Set the array name.
  void SetName(const std::string& name)
    { this->Name = name; }
  const std::string& GetName() const
    { return this->Name; }

  virtual DaxDataArray Upload(bool copy_heavy_data=false) = 0;
  virtual bool Download(const DaxDataArray& array) = 0;

protected: 
  std::string Name;

private:
  daxDisableCopyMacro(daxDataArray)
};

#endif
