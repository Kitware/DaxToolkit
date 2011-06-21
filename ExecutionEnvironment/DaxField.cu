/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __DaxField_h
#define __DaxField_h

#include "DaxCommon.h"
#include "DaxWork.cu"
#include "DaxDataObject.cu"

class DaxField
{
  DaxArray& Array;
public:
  __device__ DaxField(DaxArray& array) : Array(array)
    {
    }

  /// Set a scalar value.
  __device__ void Set(const DaxWork& work, DaxScalar scalar)
    {
    DaxArraySetterTraits::Set(work, this->Array, scalar);
    }

  __device__ DaxScalar GetScalar(const DaxWork& work)
    {
    return DaxArrayGetterTraits::GetScalar(work, this->Array);
    }
};

class DaxFieldPoint : public DaxField
{
  SUPERCLASS(DaxField);
public:
  __device__ DaxFieldPoint(DaxArray& array) : Superclass(array)
    {
    }
};

class DaxFieldCell : public DaxField
{
  SUPERCLASS(DaxField);
public:
  __device__ DaxFieldCell(DaxArray& array) : Superclass(array)
    {
    }
};


#endif
