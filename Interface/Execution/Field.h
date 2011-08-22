/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_Field_h
#define __dax_exec_Field_h

#include "Core/Execution/DataArrayTraits.h"

namespace dax { namespace exec {

class Field
{
  dax::core::DataArray& Array;
public:
  __device__ Field(dax::core::DataArray& array) : Array(array)
    {
    }

  /// Set a scalar value.
  __device__ void Set(const dax::core::exec::Work& work, dax::Scalar scalar)
    {
    dax::core::exec::DataArraySetterTraits::Set(work, this->Array, scalar);
    }

  __device__ void Set(const dax::core::exec::Work& work, dax::Vector3 vector3)
    {
    dax::core::exec::DataArraySetterTraits::Set(work, this->Array, vector3);
    }

  __device__ void Set(const dax::core::exec::Work& work, dax::Vector4 vector4)
    {
    dax::core::exec::DataArraySetterTraits::Set(work, this->Array, vector4);
    }

  __device__ dax::Scalar GetScalar(const dax::core::exec::Work& work) const
    {
    return dax::core::exec::DataArrayGetterTraits::GetScalar(work, this->Array);
    }

  __device__ dax::Vector3 GetVector3(const dax::core::exec::Work& work) const
    {
    return dax::core::exec::DataArrayGetterTraits::GetVector3(work, this->Array);
    }

  __device__ dax::Vector4 GetVector4(const dax::core::exec::Work& work) const
    {
    return dax::core::exec::DataArrayGetterTraits::GetVector4(work, this->Array);
    }
};

class FieldPoint : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldPoint(dax::core::DataArray& array) : Superclass(array)
    {
    }
};

class FieldCoordinates : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldCoordinates(dax::core::DataArray& array) : Superclass(array)
    {
    }
};

class FieldCell : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldCell(dax::core::DataArray& array) : Superclass(array)
    {
    }
};

}}

#endif
