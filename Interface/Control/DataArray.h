/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArray_h
#define __dax_cont_DataArray_h

#include "Core/Control/Object.h"

namespace dax { namespace cont {

daxDeclareClass(DataArray);

/// dax::cont::DataArray is the abstract superclass for data array object
/// containing numeric data.
class DataArray : public dax::core::cont::Object
{
public:
  DataArray();
  virtual ~DataArray();
  daxTypeMacro(DataArray, dax::core::cont::Object);

  /// Get/Set the array name.
  void SetName(const std::string& name)
    { this->Name = name; }
  const std::string& GetName() const
    { return this->Name; }

protected: 
  std::string Name;

private:
  daxDisableCopyMacro(DataArray)
};

}}

#endif
