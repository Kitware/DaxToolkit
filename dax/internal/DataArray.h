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

/// Create a dax::internal::DataArray from a raw pointer.
DAX_EXEC_CONT_EXPORT inline
DataArray<dax::Id> make_DataArrayId(dax::Id *rawData, dax::Id numEntries)
{
  return DataArray<dax::Id>(rawData, numEntries);
}

/// Create a dax::internal::DataArray from a raw pointer.
DAX_EXEC_CONT_EXPORT inline
DataArray<dax::Id3> make_DataArrayId3(dax::Id *rawData, dax::Id numTuples)
{
  return DataArray<dax::Id3>(reinterpret_cast<dax::Id3*>(rawData), numTuples);
}

/// Create a dax::internal::DataArray from a raw pointer.
DAX_EXEC_CONT_EXPORT inline
DataArray<dax::Scalar> make_DataArrayScalar(dax::Scalar *rawData,
                                            dax::Id numEntries)
{
  return DataArray<dax::Scalar>(rawData, numEntries);
}

/// Create a dax::internal::DataArray from a raw pointer.
DAX_EXEC_CONT_EXPORT inline
DataArray<dax::Vector3> make_DataArrayVector3(dax::Scalar *rawData,
                                              dax::Id numTuples)
{
  return DataArray<dax::Vector3>(reinterpret_cast<dax::Vector3*>(rawData),
                                 numTuples);
}

/// Create a dax::internal::DataArray from a raw pointer.
DAX_EXEC_CONT_EXPORT inline
DataArray<dax::Vector4> make_DataArrayVector4(dax::Scalar *rawData,
                                              dax::Id numTuples)
{
  return DataArray<dax::Vector4>(reinterpret_cast<dax::Vector4*>(rawData),
                                 numTuples);
}

}}

#endif
