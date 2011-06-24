/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataArray_h
#define __daxDataArray_h

#include "daxObject.h"

/// daxDataArray is the abstract superclass for data array object containing
/// numeric data.
class daxDataArray : public daxObject
{
public:
  daxDataArray();
  virtual ~daxDataArray();
  daxTypeMacro(daxDataArray, daxObject);

protected: 
private:
  daxDisableCopyMacro(daxDataArray)
};

/// declares daxDataArrayPtr
daxDefinePtrMacro(daxDataArray)

#endif
