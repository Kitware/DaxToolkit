/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_core_DataArray_h
#define __dax_core_DataArray_h

#include <dax/Types.h>

namespace dax { namespace internal {

/// DataArray is a basic data-storage device in Dax data model. It stores the
/// heavy data. A dataset comprises for DataArray instances assigned different
/// roles, for example an "array" for storing point coordinates, an "array" for
/// cell-connectivity etc. Different types of arrays exist. The subclasses are
/// used in control environment to define the datasets. In the execution
/// environment, user code i.e the worklet should never use DataArray directly
/// (it should rely on DaxField or subclasses). The execution environment uses
/// various traits to access raw-data from the DataArray.
class DataArray
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

  DataArray() :
    Type (UNKNOWN),
    DataType (VOID),
    RawData (0),
    SizeInBytes (0)
    {
    }

  static eDataType type(const dax::Scalar&)
    {
    return DataArray::SCALAR;
    }

  static eDataType type(const dax::Vector3&)
    {
    return DataArray::VECTOR3;
    }

  static eDataType type(const dax::Vector4&)
    {
    return DataArray::VECTOR4;
    }

  static eDataType type(const dax::Id&)
    {
    return DataArray::ID;
    }

  __device__ static int GetNumberOfComponents(const DataArray& array)
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

  DataArray& operator=(const DataArray& other)
    {
    this->Type = other.Type;
    this->DataType = other.DataType;
    this->RawData = other.RawData;
    this->SizeInBytes = other.SizeInBytes;
    return *this;
    }
};

}}

#endif
