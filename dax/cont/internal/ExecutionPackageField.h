/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __dax_cont_internal_ExecutionPackageField_h
#define __dax_cont_internal_ExecutionPackageField_h

#include <dax/Types.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorControlBadValue.h>

#include <dax/exec/Field.h>

namespace dax {
namespace cont {
namespace internal {

#define DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(super, type) \
  private: \
    typedef super< type, DeviceAdapter > Superclass; \
  public: \
    typedef typename Superclass::ValueType ValueType; \
    typedef typename Superclass::ControlArrayType ControlArrayType; \
    typedef typename Superclass::ExecutionFieldType ExecutionFieldType

namespace details {

template<class FieldT, class DeviceAdapter>
class ExecutionPackageField
{
public:
  typedef FieldT ExecutionFieldType;
  typedef typename ExecutionFieldType::ValueType ValueType;
  typedef dax::cont::ArrayHandle<ValueType, DeviceAdapter> ControlArrayType;

  ExecutionPackageField(const dax::internal::DataArray<ValueType> &array,
                        dax::Id expectedSize)
    : Field(array) {
    if (array.GetNumberOfEntries() != expectedSize)
      {
      throw dax::cont::ErrorControlBadValue(
            "Received an array that is the wrong size "
            "for the field it is intended for.");
      }
  }

  const ExecutionFieldType &GetExecutionObject() const {
    return this->Field;
  }
private:
  ExecutionFieldType Field;
};

template<class FieldT, class DeviceAdapter>
class ExecutionPackageFieldInput
    : public dax::cont::internal::details::ExecutionPackageField<FieldT, DeviceAdapter>
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(ExecutionPackageField, FieldT);
public:
  ExecutionPackageFieldInput(ControlArrayType &array,
                             dax::Id expectedSize)
    : Superclass(array.ReadyAsInput(), expectedSize), Array(array) { }

private:
  ExecutionPackageFieldInput(const ExecutionPackageFieldInput &); // Not implemented
  void operator=(const ExecutionPackageFieldInput &); // Not implemented

  ControlArrayType Array;
};

template<class FieldT, class DeviceAdapter>
class ExecutionPackageFieldOutput
    : public dax::cont::internal::details::ExecutionPackageField<FieldT, DeviceAdapter>
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(ExecutionPackageField, FieldT);
public:
  ExecutionPackageFieldOutput(ControlArrayType &array,
                              dax::Id expectedSize)
    : Superclass(array.ReadyAsOutput(), expectedSize), Array(array) { }
  ~ExecutionPackageFieldOutput() {
    this->Array.CompleteAsOutput();
  }

private:
  ExecutionPackageFieldOutput(const ExecutionPackageFieldOutput &); // Not implemented
  void operator=(const ExecutionPackageFieldOutput &);  // Not implemented

  ControlArrayType Array;
};

} // namespace details

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldPointInput
    : public dax::cont::internal::details::ExecutionPackageFieldInput<dax::exec::FieldPoint<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldInput,
                                        dax::exec::FieldPoint<T>);
public:
  template<class GridT>
  ExecutionPackageFieldPointInput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                                  const GridT &grid)
    : Superclass(array, grid.GetNumberOfPoints()) { }
};

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldPointOutput
    : public dax::cont::internal::details::ExecutionPackageFieldOutput<dax::exec::FieldPoint<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldOutput,
                                        dax::exec::FieldPoint<T>);
public:
  template<class GridT>
  ExecutionPackageFieldPointOutput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                                   const GridT &grid)
    : Superclass(array, grid.GetNumberOfPoints()) { }
};

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldCellInput
    : public dax::cont::internal::details::ExecutionPackageFieldInput<dax::exec::FieldCell<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldInput,
                                        dax::exec::FieldCell<T>);
public:
  template<class GridT>
  ExecutionPackageFieldCellInput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                                 const GridT &grid)
    : Superclass(array, grid.GetNumberOfCells()) { }
};

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldCellOutput
    : public dax::cont::internal::details::ExecutionPackageFieldOutput<dax::exec::FieldCell<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldOutput,
                                        dax::exec::FieldCell<T>);
public:
  template<class GridT>
  ExecutionPackageFieldCellOutput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                                  const GridT &grid)
    : Superclass(array, grid.GetNumberOfCells()) { }
};

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldInput
    : public dax::cont::internal::details::ExecutionPackageFieldInput<dax::exec::Field<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldInput,
                                        dax::exec::Field<T>);
public:
  ExecutionPackageFieldInput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                             dax::Id expectedSize)
    : Superclass(array, expectedSize) { }
};

template<typename T, class DeviceAdapter>
class ExecutionPackageFieldOutput
    : public dax::cont::internal::details::ExecutionPackageFieldOutput<dax::exec::Field<T>, DeviceAdapter >
{
  DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE(details::ExecutionPackageFieldOutput,
                                        dax::exec::Field<T>);
public:
  ExecutionPackageFieldOutput(dax::cont::ArrayHandle<T, DeviceAdapter> &array,
                             dax::Id expectedSize)
    : Superclass(array, expectedSize) { }
};

template<class GridT, class DeviceAdapter>
class ExecutionPackageFieldCoordinatesInput
{
public:
  typedef dax::Vector3 ValueType;
  typedef dax::exec::FieldCoordinates ExecutionFieldType;

  ExecutionPackageFieldCoordinatesInput(typename GridT::Points) { }

  ExecutionFieldType GetExecutionObject() const {
    dax::internal::DataArray<dax::Vector3> dummyArray;
    dax::exec::FieldCoordinates field(dummyArray);
    return field;
  }
};

#undef DAX_EXECUTION_PACKAGE_FIELD_SUPERTYPE

}
}
}

#endif //__dax_cont_internal_ExecutionPackageField_h
