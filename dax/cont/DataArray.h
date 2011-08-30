/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_cont_DataArray_h
#define __dax_cont_DataArray_h

#include <dax/cont/internal/Object.h>

namespace dax { namespace cont {

daxDeclareClass(DataArray);

/// dax::internal::DataArray is the abstract superclass for data array object
/// containing numeric data.
class DataArray : public dax::cont::internal::Object
{
public:
  DataArray();
  virtual ~DataArray();
  daxTypeMacro(DataArray, dax::cont::internal::Object);

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
