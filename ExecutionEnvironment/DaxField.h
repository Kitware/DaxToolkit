/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __DaxField_h
#define __DaxField_h

#include "DaxDataArrayTraits.h"
class DaxWork;

class DaxField
{
  DaxDataArray& Array;
public:
  __device__ DaxField(DaxDataArray& array) : Array(array)
    {
    }

  /// Set a scalar value.
  __device__ void Set(const DaxWork& work, DaxScalar scalar)
    {
    DaxDataArraySetterTraits::Set(work, this->Array, scalar);
    }

  __device__ void Set(const DaxWork& work, DaxVector3 vector3)
    {
    DaxDataArraySetterTraits::Set(work, this->Array, vector3);
    }

  __device__ void Set(const DaxWork& work, DaxVector4 vector4)
    {
    DaxDataArraySetterTraits::Set(work, this->Array, vector4);
    }

  __device__ DaxScalar GetScalar(const DaxWork& work) const
    {
    return DaxDataArrayGetterTraits::GetScalar(work, this->Array);
    }

  __device__ DaxVector3 GetVector3(const DaxWork& work) const
    {
    return DaxDataArrayGetterTraits::GetVector3(work, this->Array);
    }

  __device__ DaxVector4 GetVector4(const DaxWork& work) const
    {
    return DaxDataArrayGetterTraits::GetVector4(work, this->Array);
    }
};

class DaxFieldPoint : public DaxField
{
  SUPERCLASS(DaxField);
public:
  __device__ DaxFieldPoint(DaxDataArray& array) : Superclass(array)
    {
    }
};

class DaxFieldCoordinates : public DaxField
{
  SUPERCLASS(DaxField);
public:
  __device__ DaxFieldCoordinates(DaxDataArray& array) : Superclass(array)
    {
    }
};

class DaxFieldCell : public DaxField
{
  SUPERCLASS(DaxField);
public:
  __device__ DaxFieldCell(DaxDataArray& array) : Superclass(array)
    {
    }
};


#endif
