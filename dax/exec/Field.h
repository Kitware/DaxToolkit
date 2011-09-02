/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_Field_h
#define __dax_exec_Field_h

#include <dax/internal/DataArray.h>

namespace dax { namespace exec {

template<typename T>
class Field
{
public:
  typedef T ValueType;

  DAX_EXEC_EXPORT Field(dax::internal::DataArray<ValueType> &array)
    : Array(array)
  {
  }

  /// Get the internal array.  Work objects use this to get/set values.
  DAX_EXEC_EXPORT const dax::internal::DataArray<ValueType> &GetArray() const
  {
    return this->Array;
  }
  DAX_EXEC_EXPORT dax::internal::DataArray<ValueType> &GetArray()
  {
    return this->Array;
  }

private:
  dax::internal::DataArray<ValueType> &Array;
};

template<typename T>
class FieldPoint : public dax::exec::Field<T>
{
public:
  typedef T ValueType;
  typedef dax::exec::Field<ValueType> Superclass;

  DAX_EXEC_EXPORT FieldPoint(dax::internal::DataArray<ValueType> &array)
    : Superclass(array)
  { }
};

class FieldCoordinates : public dax::exec::FieldPoint<dax::Vector3>
{
public:
  typedef dax::Vector3 ValueType;
  typedef dax::exec::FieldPoint<ValueType> Superclass;
  DAX_EXEC_EXPORT FieldCoordinates(dax::internal::DataArray<ValueType> &array)
    : Superclass(array)
   { }
};

template<class T>
class FieldCell : public dax::exec::Field<T>
{
public:
  typedef T ValueType;
  typedef dax::exec::Field<ValueType> Superclass;
  DAX_EXEC_EXPORT FieldCell(dax::internal::DataArray<ValueType> &array)
    : Superclass(array)
  { }
};

}}

#endif
