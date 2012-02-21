/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_internal_DataArray_h
#define __dax_internal_DataArray_h

#include <dax/internal/ExportMacros.h>

#include <dax/Types.h>

//------------------------------------------------------------------------------
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


  /// Builds a DataArray with no data
  DAX_EXEC_CONT_EXPORT DataArray()
    : Data(0), NumEntries(0), IndexOffset(0)
  { }

  /// Builds a DataArray with the given C pointer.
  DAX_EXEC_CONT_EXPORT DataArray(ValueType *data,
                                 dax::Id numEntries)
    : Data(data), NumEntries(numEntries), IndexOffset(0)
  { }

  /// Builds a DataArray with the given C pointer and an offset.
  DAX_EXEC_CONT_EXPORT DataArray(ValueType *data,
                                 dax::Id numEntries,
                                 dax::Id indexOffset)
    : Data(data), NumEntries(numEntries), IndexOffset(indexOffset)
  { }

  /// Returns the value in the array at the given index.  The index is
  /// adjusted by the IndexOffset.
  DAX_EXEC_CONT_EXPORT const ValueType &GetValue(dax::Id index) const
  {
    return this->Data[index-this->IndexOffset];
  }

  /// Sets the value in the array at the given index.  The index is
  /// adjusted by the IndexOffset.
  DAX_EXEC_CONT_EXPORT void SetValue(dax::Id index, const ValueType &value)
  {
    this->Data[index-this->IndexOffset] = value;
  }

  /// Returns the number of entries in the managed array.  The valid indices
  /// are from IndexOffset to NumberOfEntries+IndexOffset-1.
  DAX_EXEC_CONT_EXPORT dax::Id GetNumberOfEntries() const
  {
    return this->NumEntries;
  }

  /// Returns the offset from indices given this object and indices in the C
  /// array. This offset is convienient for creating per-thread intermediate
  /// arrays that contain only some small subset of the whole array.
  DAX_EXEC_CONT_EXPORT dax::Id GetIndexOffset() const
  {
    return this->IndexOffset;
  }

  /// Returns the C array held by this DataArray.  Don't forget that the
  /// indices in this array may be offset in the DataArray by IndexOffset.
  DAX_EXEC_CONT_EXPORT const ValueType *GetPointer() const
  {
    return this->Data;
  }
  DAX_EXEC_CONT_EXPORT ValueType *GetPointer() { return this->Data; }

  /// Sets the array used in the DataArray.  The pointer, its size, and
  /// the offset are given.
  DAX_EXEC_CONT_EXPORT void SetPointer(ValueType *data,
                                       dax::Id numEntries,
                                       dax::Id indexOffset = 0)
  {
    this->Data = data;
    this->NumEntries = numEntries;
    this->IndexOffset = indexOffset;
  }

private:
  ValueType *Data;
  dax::Id NumEntries;
  dax::Id IndexOffset;
};

}}

#endif
