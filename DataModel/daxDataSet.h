/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataSet_h
#define __daxDataSet_h

#include "daxObject.h"
#include <vector>

#ifndef SKIP_DOXYGEN
class daxDataArray;
daxDefinePtrMacro(daxDataArray);
#endif

/// daxDataSet is the abstract superclass for data array object containing
/// numeric data.
class daxDataSet : public daxObject
{
public:
  daxTypeMacro(daxDataSet, daxObject);

  std::vector<daxDataArrayPtr> PointData;
  std::vector<daxDataArrayPtr> CellData;
  virtual daxDataArrayPtr GetPointCoordinates() const = 0;
  virtual daxDataArrayPtr GetCellArray() const = 0;

protected:
  daxDataSet();
  virtual ~daxDataSet();

private:
  daxDisableCopyMacro(daxDataSet)
};

/// declares daxDataSetPtr
daxDefinePtrMacro(daxDataSet)

#endif
