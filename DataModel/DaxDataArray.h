/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArray_h
#define __DaxDataArray_h

#include "daxTypes.h"

class DaxDataArray
{
public:
  enum eType
    {
    UNKNOWN= -1,
    IRREGULAR = 1,
    STRUCTURED_POINTS = 2,
    STRUCTURED_CONNECTIVITY=3,
    };
  enum eDataType
    {
    VOID,
    SCALAR,
    VECTOR3,
    VECTOR4,
    ID
    };

  eType Type;
  eDataType DataType;
  void* RawData;
  unsigned int SizeInBytes;

  __device__ __host__ DaxDataArray() :
    Type (UNKNOWN),
    DataType (VOID),
    RawData (NULL),
    SizeInBytes (0)
    {
    }

  __host__ static eDataType type(const DaxScalar&)
    {
    return DaxDataArray::SCALAR;
    }

  __host__ static eDataType type(const DaxVector3&)
    {
    return DaxDataArray::VECTOR3;
    }

  __host__ static eDataType type(const DaxVector4&)
    {
    return DaxDataArray::VECTOR4;
    }

  __host__ static eDataType type(const DaxId&)
    {
    return DaxDataArray::ID;
    }
};

#endif
