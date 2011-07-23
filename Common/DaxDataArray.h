/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __DaxDataArray_h
#define __DaxDataArray_h

#include "daxTypes.h"

/// DaxDataArray is a basic data-storage device in Dax data model. It stores the
/// heavy data. A dataset comprises for DaxDataArray instances assigned different
/// roles, for example an "array" for storing point coordinates, an "array" for
/// cell-connectivity etc. Different types of arrays exist. The subclasses are
/// used in control environment to define the datasets. In the execution
/// environment, user code i.e the worklet should never use DaxDataArray directly
/// (it should rely on DaxField or subclasses). The execution environment uses
/// various traits to access raw-data from the DaxDataArray.
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

#ifdef __CUDACC__
  __device__ static int GetNumberOfComponents(const DaxDataArray& array)
    {
    int num_comps;
    switch (array.DataType)
      {
    case SCALAR:
      num_comps = 1;
      break;
    case VECTOR3:
      num_comps = 3;
      break;
    case VECTOR4:
      num_comps = 4;
      break;
    default:
      num_comps = 1;
      }
    return num_comps;
    }
#endif
};

#endif
