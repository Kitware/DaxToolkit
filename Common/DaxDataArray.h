/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArray_h
#define __DaxDataArray_h

#include "daxTypes.h"

/// DaxDataArray is the data-structure used to represent a data-array in the
/// Execution environment. In the control environment, users never use this
/// class directly. It's only meant to be used by the DataModel classes to
/// upload/download data to/from the execution environment.
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

  DaxDataArray() :
    Type (UNKNOWN),
    DataType (VOID),
    RawData (NULL),
    SizeInBytes (0)
    {
    }

  static eDataType type(const DaxScalar&)
    {
    return DaxDataArray::SCALAR;
    }

  static eDataType type(const DaxVector3&)
    {
    return DaxDataArray::VECTOR3;
    }

  static eDataType type(const DaxVector4&)
    {
    return DaxDataArray::VECTOR4;
    }

  static eDataType type(const DaxId&)
    {
    return DaxDataArray::ID;
    }
};

#endif
