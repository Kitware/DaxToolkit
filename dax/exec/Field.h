/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_Field_h
#define __dax_exec_Field_h

#include <dax/exec/internal/DataArrayTraits.h>

namespace dax { namespace exec {

class Field
{
  dax::internal::DataArray& Array;
public:
  __device__ Field(dax::internal::DataArray& array) : Array(array)
    {
    }

  /// Set a scalar value.
  __device__ void Set(const dax::exec::Work& work, dax::Scalar scalar)
    {
    dax::exec::internal::DataArraySetterTraits::Set(work, this->Array, scalar);
    }

  __device__ void Set(const dax::exec::Work& work, dax::Vector3 vector3)
    {
    dax::exec::internal::DataArraySetterTraits::Set(work, this->Array, vector3);
    }

  __device__ void Set(const dax::exec::Work& work, dax::Vector4 vector4)
    {
    dax::exec::internal::DataArraySetterTraits::Set(work, this->Array, vector4);
    }

  __device__ dax::Scalar GetScalar(const dax::exec::Work& work) const
    {
    return dax::exec::internal::DataArrayGetterTraits::GetScalar(work, this->Array);
    }

  __device__ dax::Vector3 GetVector3(const dax::exec::Work& work) const
    {
    return dax::exec::internal::DataArrayGetterTraits::GetVector3(work, this->Array);
    }

  __device__ dax::Vector4 GetVector4(const dax::exec::Work& work) const
    {
    return dax::exec::internal::DataArrayGetterTraits::GetVector4(work, this->Array);
    }
};

class FieldPoint : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldPoint(dax::internal::DataArray& array) : Superclass(array)
    {
    }
};

class FieldCoordinates : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldCoordinates(dax::internal::DataArray& array) : Superclass(array)
    {
    }
};

class FieldCell : public dax::exec::Field
{
  SUPERCLASS(dax::exec::Field);
public:
  __device__ FieldCell(dax::internal::DataArray& array) : Superclass(array)
    {
    }
};

}}

#endif
