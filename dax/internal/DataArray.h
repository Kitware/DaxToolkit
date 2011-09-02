/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_internal_DataArray_h
#define __dax_internal_DataArray_h

#include <dax/internal/ExportMacros.h>

#include <dax/Types.h>

namespace dax { namespace internal {

/// DataArary is a simple but basic data-storage device in the Dax data model.
/// It stores an array of some templated type that should be a compact
/// storage structure.  No management of the array is done other than hold
/// it.
template<typename T>
class DataArray
{
public:
  typedef T ValueType;

  DAX_EXEC_CONT_EXPORT DataArray(ValueType *data, dax::Id numEntries)
    : Data(data), NumEntries(numEntries)
  { }

  DAX_EXEC_CONT_EXPORT DataArray() : Data(0), NumEntries(0) { }

  DAX_EXEC_CONT_EXPORT const ValueType &GetValue(dax::Id index) const
  {
    return this->Data[index];
  }

  DAX_EXEC_CONT_EXPORT void SetValue(dax::Id index, const ValueType &value)
  {
    this->Data[index] = value;
  }

  DAX_EXEC_CONT_EXPORT dax::Id GetNumberOfEntries() const
  {
    return this->NumEntries;
  }

  DAX_EXEC_CONT_EXPORT const ValueType *GetPointer() const
  {
    return this->Data;
  }
  DAX_EXEC_CONT_EXPORT ValueType *GetPointer() { return this->Data; }

  DAX_EXEC_CONT_EXPORT void SetPointer(ValueType *data, dax::Id numEntries)
  {
    this->Data = data;
    this->NumEntries = numEntries;
  }

private:
  ValueType *Data;
  dax::Id NumEntries;
};

}}

#endif
