/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataSet_h
#define __dax_cont_DataSet_h

#include "Core/Control/Object.h"
#include <vector>

namespace dax { namespace cont {

daxDeclareClass(DataArray);
daxDeclareClass(DataSet);

/// daxDataSet is the abstract superclass for data array object containing
/// numeric data.
class DataSet : public dax::core::cont::Object
{
public:
  DataSet();
  virtual ~DataSet();

  daxTypeMacro(DataSet, dax::core::cont::Object);

  std::vector<dax::cont::DataArrayPtr> PointData;
  std::vector<dax::cont::DataArrayPtr> CellData;

  /// Provides access to the point-coordinates array.
  virtual dax::cont::DataArrayPtr GetPointCoordinates() const = 0;

  /// Provides access to the cell-array.
  virtual dax::cont::DataArrayPtr GetCellArray() const = 0;

private:
  daxDisableCopyMacro(DataSet)
};

}}

#endif
